"""Graphe agentique SANAD (LangGraph) — Corrective RAG avec auto-vérification.

    guard ─► normalize ─► route ─► retrieve ─► grade ─┬─► generate ─► check_grounding ─┬─► checklist ─► END
      │                     │          ▲               │        ▲                        │
      ▼                     ▼          └── rewrite ◄───┤        └──── (regenerate) ◄─────┘
    END (blocked)     END (hors domaine)               └─► END (introuvable)
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import settings
from .derja import detect_language, gloss
from .guardrails import check_input, cited_ids, mask_pii, refusal, strip_invalid_citations
from .retriever import HybridRetriever
from .schemas import (AskResponse, Checklist, DocGrades, DomainRoute, GroundingCheck,
                      RewrittenQuery, Source)


class SanadState(TypedDict, total=False):
    question: str
    history: list[dict]
    want_checklist: bool
    language: str
    normalized_query: str
    intent: str
    docs: list[Document]
    relevant: list[Document]
    rewrites: int
    regenerations: int
    unsupported: list[str]
    answer: str
    grounding: float | None
    status: str
    checklist: dict | None
    trace: Annotated[list[str], operator.add]


def _format_sources(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({d.metadata.get('title', '')}) {d.page_content}" for i, d in enumerate(docs, 1))


def _history_text(history: list[dict], n: int = 4) -> str:
    turns = history[-n:]
    return "\n".join(f"{t['role']}: {t['content'][:400]}" for t in turns)


class SanadGraph:
    """Construit le graphe ; LLM et retriever sont injectés (testable sans Ollama)."""

    def __init__(self, llm, retriever: HybridRetriever, top_k: int | None = None):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k or settings.top_k
        self.graph = self._build()

    # ---------- helpers LLM ----------
    def _structured(self, schema, system: str, user: str):
        return self.llm.with_structured_output(schema).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)])

    # ---------- nœuds ----------
    def guard(self, s: SanadState) -> dict:
        q = s["question"]
        lang = detect_language(q)
        res = check_input(q, settings.max_question_chars)
        if not res.allowed:
            return {"language": lang, "status": "blocked", "answer": refusal(res.reason, lang),
                    "trace": [f"🛡️ Garde-fou : requête bloquée ({res.reason})"]}
        return {"language": lang, "question": mask_pii(q), "rewrites": 0, "regenerations": 0,
                "trace": [f"🛡️ Garde-fou : OK — langue détectée : {lang}"]}

    def normalize(self, s: SanadState) -> dict:
        q, lang, history = s["question"], s["language"], s.get("history") or []
        if lang == "fr" and not history:
            return {"normalized_query": q, "trace": ["✏️ Normalisation : inutile (français, sans historique)"]}
        hint = f"\nGlossaire mot-à-mot (indice) : {gloss(q)}" if lang == "derja" else ""
        ctx = f"Historique :\n{_history_text(history)}\n\n" if history else ""
        try:
            out = self._structured(RewrittenQuery, prompts.NORMALIZE, f"{ctx}Question : {q}{hint}")
            nq = out.query.strip() or q
        except Exception as e:  # repli déterministe si le LLM échoue
            nq = gloss(q) if lang == "derja" else q
            return {"normalized_query": nq, "trace": [f"✏️ Normalisation (repli glossaire, {type(e).__name__}) : {nq}"]}
        return {"normalized_query": nq, "trace": [f"✏️ Normalisation : « {nq} »"]}

    def route(self, s: SanadState) -> dict:
        try:
            r: DomainRoute = self._structured(DomainRoute, prompts.ROUTE, s["normalized_query"])
        except Exception:
            return {"intent": "information", "trace": ["🧭 Routage : repli (supposé dans le domaine)"]}
        if not r.in_domain:
            return {"status": "out_of_domain", "answer": refusal("out_of_domain", s["language"]),
                    "trace": [f"🧭 Routage : hors domaine — {r.reason}"]}
        return {"intent": r.intent, "trace": [f"🧭 Routage : dans le domaine (intention : {r.intent})"]}

    def retrieve(self, s: SanadState) -> dict:
        extra = [s["question"]] if s["question"] != s["normalized_query"] else []
        hits = self.retriever.search(s["normalized_query"], k=self.top_k, extra_queries=extra)
        titles = sorted({h.doc.metadata.get("title", "?") for h in hits})
        return {"docs": [h.doc for h in hits],
                "trace": [f"🔎 Recherche hybride BM25+FAISS : {len(hits)} extraits ({', '.join(titles)})"]}

    def grade(self, s: SanadState) -> dict:
        docs = s.get("docs") or []
        if not docs:
            return {"relevant": [], "trace": ["⚖️ Évaluation : aucun extrait"]}
        user = f"Question : {s['normalized_query']}\n\nExtraits :\n{_format_sources(docs)}"
        try:
            g: DocGrades = self._structured(DocGrades, prompts.GRADE, user)
            relevant = [docs[i - 1] for i in dict.fromkeys(g.relevant_ids) if 1 <= i <= len(docs)]
        except Exception:
            relevant = docs
        return {"relevant": relevant, "trace": [f"⚖️ Évaluation de pertinence : {len(relevant)}/{len(docs)} extraits retenus"]}

    def rewrite(self, s: SanadState) -> dict:
        try:
            out = self._structured(RewrittenQuery, prompts.REWRITE,
                                   f"Question originale : {s['question']}\nRequête utilisée : {s['normalized_query']}")
            nq = out.query.strip()
        except Exception:
            nq = s["normalized_query"]
        n = s.get("rewrites", 0) + 1
        return {"normalized_query": nq, "rewrites": n, "trace": [f"🔁 Réécriture #{n} : « {nq} »"]}

    def not_found(self, s: SanadState) -> dict:
        return {"status": "not_found", "answer": refusal("not_found", s["language"]),
                "trace": ["🚫 Aucune source pertinente : réponse refusée plutôt qu'inventée"]}

    def generate(self, s: SanadState) -> dict:
        docs = s["relevant"]
        system = prompts.GENERATE.format(language_instruction=prompts.LANGUAGE_INSTRUCTIONS[s["language"]])
        if s.get("unsupported"):
            system += prompts.GENERATE_STRICT_SUFFIX.format(claims="\n".join(f"- {c}" for c in s["unsupported"]))
        user = f"Sources :\n{_format_sources(docs)}\n\nQuestion : {s['question']}"
        if s["normalized_query"] != s["question"]:
            user += f"\n(Reformulation : {s['normalized_query']})"
        msg = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        answer = strip_invalid_citations(str(msg.content).strip(), set(range(1, len(docs) + 1)))
        label = "Régénération stricte" if s.get("unsupported") else "Génération"
        return {"answer": answer, "trace": [f"🧠 {label} avec {len(docs)} sources citées"]}

    def check_grounding(self, s: SanadState) -> dict:
        user = f"SOURCES :\n{_format_sources(s['relevant'])}\n\nRÉPONSE :\n{s['answer']}"
        try:
            g: GroundingCheck = self._structured(GroundingCheck, prompts.GROUNDING, user)
        except Exception:
            return {"grounding": None, "unsupported": [], "trace": ["✅ Vérification : indisponible"]}
        update: dict[str, Any] = {"grounding": round(g.score, 2)}
        if g.grounded or not g.unsupported_claims:
            update.update(unsupported=[], trace=[f"✅ Vérification des faits : score {g.score:.2f}"])
        else:
            update.update(unsupported=g.unsupported_claims,
                          regenerations=s.get("regenerations", 0) + 1,
                          trace=[f"⚠️ Vérification : {len(g.unsupported_claims)} affirmation(s) non appuyée(s), score {g.score:.2f}"])
        return update

    def finalize(self, s: SanadState) -> dict:
        update: dict[str, Any] = {"status": "answered"}
        answer = s["answer"]
        if s.get("unsupported"):  # toujours non appuyée après régénération
            warn = ("⚠️ Certaines informations n'ont pas pu être vérifiées dans les sources."
                    if s["language"] != "ar" else "⚠️ بعض المعلومات لم يتم التحقق منها في المصادر.")
            answer = f"{answer}\n\n{warn}"
        if not cited_ids(answer):
            answer += "\n\n(Sources : " + ", ".join(dict.fromkeys(d.metadata.get("title", "") for d in s["relevant"])) + ")"
        update["answer"] = answer
        if s.get("want_checklist"):
            lang = "arabe" if s["language"] == "ar" else "français"
            user = f"Sources :\n{_format_sources(s['relevant'])}\n\nRéponse :\n{answer}"
            try:
                c: Checklist = self._structured(Checklist, prompts.CHECKLIST.format(language=lang), user)
                update["checklist"] = c.model_dump()
                update["trace"] = ["📋 Fiche démarche structurée générée (Pydantic)"]
            except Exception as e:
                update["trace"] = [f"📋 Fiche démarche : échec ({type(e).__name__})"]
        return update

    # ---------- routage conditionnel ----------
    @staticmethod
    def _after_guard(s: SanadState) -> str:
        return END if s.get("status") == "blocked" else "normalize"

    @staticmethod
    def _after_route(s: SanadState) -> str:
        return END if s.get("status") == "out_of_domain" else "retrieve"

    def _after_grade(self, s: SanadState) -> str:
        if s.get("relevant"):
            return "generate"
        return "rewrite" if s.get("rewrites", 0) < settings.max_rewrites else "not_found"

    @staticmethod
    def _after_check(s: SanadState) -> str:
        if s.get("unsupported") and s.get("regenerations", 0) <= settings.max_regenerations:
            return "generate"
        return "finalize"

    def _build(self):
        g = StateGraph(SanadState)
        for name in ("guard", "normalize", "route", "retrieve", "grade", "rewrite",
                     "not_found", "generate", "check_grounding", "finalize"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "guard")
        g.add_conditional_edges("guard", self._after_guard, ["normalize", END])
        g.add_edge("normalize", "route")
        g.add_conditional_edges("route", self._after_route, ["retrieve", END])
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, ["generate", "rewrite", "not_found"])
        g.add_edge("rewrite", "retrieve")
        g.add_edge("not_found", END)
        g.add_edge("generate", "check_grounding")
        g.add_conditional_edges("check_grounding", self._after_check, ["generate", "finalize"])
        g.add_edge("finalize", END)
        return g.compile()

    # ---------- API publique ----------
    def run(self, question: str, history: list[dict] | None = None, checklist: bool = False) -> SanadState:
        """Exécute le graphe et retourne l'état brut (utilisé par l'évaluation)."""
        return self.graph.invoke({"question": question, "history": history or [], "want_checklist": checklist,
                                  "trace": []}, {"recursion_limit": 30})

    def ask(self, question: str, history: list[dict] | None = None, checklist: bool = False) -> AskResponse:
        t0 = time.perf_counter()
        s = self.run(question, history, checklist)
        sources = []
        if s.get("status") == "answered":
            for i, d in enumerate(s.get("relevant") or [], 1):
                body = d.page_content.split("\n", 1)[-1]
                sources.append(Source(id=i, title=d.metadata.get("title", ""), section=d.metadata.get("section", ""),
                                      source=d.metadata.get("source", ""), snippet=body[:300]))
        return AskResponse(
            answer=s.get("answer", ""), status=s.get("status", "not_found"), language=s.get("language", "fr"),
            normalized_query=s.get("normalized_query", question), sources=sources,
            grounding_score=s.get("grounding"), checklist=s.get("checklist"),
            trace=s.get("trace", []), latency_s=round(time.perf_counter() - t0, 2))


def load_agent(index_dir=None) -> SanadGraph:
    """Instancie l'agent de production (Ollama + index local)."""
    from .llm import get_llm

    return SanadGraph(get_llm(), HybridRetriever.load(index_dir))


def mermaid() -> str:
    """Diagramme du graphe (pour le README)."""
    return SanadGraph.__new__(SanadGraph)._build().get_graph().draw_mermaid()
