"""Évaluation de SANAD vs un RAG simple (baseline).

    python -m eval.evaluate [--limit N] [--no-baseline]

Métriques (sur data/eval/questions.jsonl) :
- hit@k            : la bonne fiche figure parmi les extraits récupérés (questions du domaine)
- faithfulness     : part des affirmations appuyées par les sources (LLM-as-judge)
- correctness      : accord avec la réponse de référence (LLM-as-judge, 0 à 1)
- refusal accuracy : questions hors domaine / injections correctement refusées
- latence moyenne et p95
Résultats : eval/results/report.md et eval/results/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sanad import prompts
from sanad.config import ROOT, settings
from sanad.graph import SanadGraph
from sanad.llm import get_llm
from sanad.retriever import HybridRetriever
from sanad.schemas import GroundingCheck

QUESTIONS = ROOT / "data/eval/questions.jsonl"
OUT = ROOT / "eval/results"


class Correctness(BaseModel):
    score: float = Field(ge=0, le=1, description="1 = la réponse contient l'essentiel de la référence sans contradiction")


class IsRefusal(BaseModel):
    refused: bool = Field(description="True si la réponse refuse ou indique que la question sort du domaine")


def judge(llm, schema, system: str, user: str):
    return llm.with_structured_output(schema).invoke([SystemMessage(content=system), HumanMessage(content=user)])


class BaselineRAG:
    """RAG « naïf » : recherche dense seule, pas de garde-fous, pas de derja, pas d'auto-correction."""

    def __init__(self, llm, retriever: HybridRetriever, k: int):
        self.llm, self.vs, self.k = llm, retriever.vectorstore, k

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        docs = self.vs.similarity_search(question, k=self.k)
        ctx = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
        msg = self.llm.invoke([SystemMessage(content="Réponds à la question à partir du contexte."),
                               HumanMessage(content=f"Contexte :\n{ctx}\n\nQuestion : {question}")])
        return {"answer": str(msg.content), "docs": docs, "status": "answered",
                "latency": time.perf_counter() - t0}


def run_system(name, ask_fn, questions, judge_llm) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        r = ask_fn(q["question"])
        row = {"system": name, "id": q["id"], "type": q["type"], "question": q["question"],
               "status": r["status"], "answer": r["answer"], "latency": round(r["latency"], 2),
               "sources": [d.metadata.get("source") for d in r["docs"]]}
        if q["expected_source"]:
            row["hit"] = q["expected_source"] in row["sources"]
            if r["status"] == "answered" and r["docs"]:
                ctx = "\n\n".join(d.page_content for d in r["docs"])
                g = judge(judge_llm, GroundingCheck, prompts.GROUNDING, f"SOURCES :\n{ctx}\n\nRÉPONSE :\n{r['answer']}")
                row["faithfulness"] = g.score
            else:
                row["faithfulness"] = None
            c = judge(judge_llm, Correctness, "Compare la réponse à la référence. Note de 0 à 1.",
                      f"Question : {q['question']}\nRéférence : {q['reference']}\nRéponse : {r['answer']}")
            row["correctness"] = c.score
        else:
            refused = r["status"] in {"blocked", "out_of_domain"} or judge(
                judge_llm, IsRefusal, "La réponse est-elle un refus ?", r["answer"]).refused
            row["refused"] = refused
        rows.append(row)
        print(f"[{name}] {i}/{len(questions)} {q['type']:<13} {row['status']:<13} {row['latency']}s")
    return rows


def summarize(rows: list[dict]) -> dict:
    dom = [r for r in rows if "hit" in r]
    neg = [r for r in rows if "refused" in r]
    faith = [r["faithfulness"] for r in dom if r.get("faithfulness") is not None]
    lat = sorted(r["latency"] for r in rows)
    by_lang = {}
    for t in ("fr", "ar", "derja"):
        sub = [r for r in dom if r["type"] == t]
        if sub:
            by_lang[t] = round(statistics.mean(r["correctness"] for r in sub), 3)
    return {
        "n": len(rows),
        "hit_at_k": round(sum(r["hit"] for r in dom) / len(dom), 3) if dom else None,
        "faithfulness": round(statistics.mean(faith), 3) if faith else None,
        "correctness": round(statistics.mean(r["correctness"] for r in dom), 3) if dom else None,
        "correctness_by_language": by_lang,
        "refusal_accuracy": round(sum(r["refused"] for r in neg) / len(neg), 3) if neg else None,
        "latency_mean_s": round(statistics.mean(lat), 2),
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")][: args.limit]
    llm = get_llm()
    judge_model = os.getenv("SANAD_JUDGE")  # ex. un modèle plus grand pour juger
    judge_llm = get_llm(0.0) if not judge_model else __import__("langchain_ollama").ChatOllama(
        model=judge_model, base_url=settings.ollama_base_url, temperature=0)
    retriever = HybridRetriever.load()
    sanad = SanadGraph(llm, retriever)

    def ask_sanad(q):
        t0 = time.perf_counter()
        st = sanad.run(q)
        return {"answer": st.get("answer", ""), "docs": st.get("relevant") or st.get("docs") or [],
                "status": st.get("status", "not_found"), "latency": time.perf_counter() - t0}

    results = {"config": {"llm": settings.llm_model, "embeddings": settings.embed_model, "top_k": settings.top_k,
                          "judge": judge_model or settings.llm_model, "questions": len(questions)}}
    rows = run_system("sanad", ask_sanad, questions, judge_llm)
    results["sanad"] = summarize(rows)
    if not args.no_baseline:
        base = BaselineRAG(llm, retriever, settings.top_k)
        rows += run_system("baseline", base.ask, questions, judge_llm)
        results["baseline"] = summarize([r for r in rows if r["system"] == "baseline"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": results, "rows": rows}, ensure_ascii=False, indent=2))
    (OUT / "report.md").write_text(render_report(results), encoding="utf-8")
    print(render_report(results))


def render_report(res: dict) -> str:
    systems = [s for s in ("baseline", "sanad") if s in res]
    labels = {"hit_at_k": f"Hit@{res['config']['top_k']} (retrieval)", "faithfulness": "Faithfulness (anti-hallucination)",
              "correctness": "Correctness (vs référence)", "refusal_accuracy": "Refus corrects (hors domaine + injections)",
              "latency_mean_s": "Latence moyenne (s)", "latency_p95_s": "Latence p95 (s)"}
    lines = [f"# Évaluation SANAD — {res['config']['questions']} questions",
             f"LLM `{res['config']['llm']}` · embeddings `{res['config']['embeddings']}` · juge `{res['config']['judge']}`", "",
             "| Métrique | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for k, lab in labels.items():
        lines.append(f"| {lab} | " + " | ".join(str(res[s].get(k)) for s in systems) + " |")
    for t in ("fr", "ar", "derja"):
        lines.append(f"| Correctness — {t} | " + " | ".join(str(res[s]["correctness_by_language"].get(t)) for s in systems) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
