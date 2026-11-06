"""LLM simulé : permet de tester tout le graphe sans Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanad.config import settings  # noqa: E402
from sanad.ingest import load_documents, split_documents  # noqa: E402
from sanad.retriever import HybridRetriever  # noqa: E402
from sanad.schemas import (Checklist, DocGrades, DomainRoute, GroundingCheck,  # noqa: E402
                           RewrittenQuery)


class FakeLLM:
    """Répond selon le schéma demandé ; les réponses sont configurables par test."""

    def __init__(self, **overrides):
        self.calls: list[str] = []
        self.answers = list(overrides.pop("answers", ["Il faut la CIN et des photos [1]. Timbre fiscal à vérifier [2]."]))
        self.grounding = list(overrides.pop("grounding", [GroundingCheck(grounded=True, score=0.95)]))
        self.o = overrides

    def with_structured_output(self, schema):
        llm = self

        class _Runner:
            def invoke(self, messages):
                llm.calls.append(schema.__name__)
                user = messages[-1].content
                if schema is DomainRoute:
                    return llm.o.get("route", DomainRoute(in_domain=True, intent="procedure", reason="démarche"))
                if schema is RewrittenQuery:
                    if "normalize" in llm.o:
                        return RewrittenQuery(query=llm.o["normalize"])
                    return RewrittenQuery(query=user.split("Question :")[-1].split("\n")[0].strip() or "requête")
                if schema is DocGrades:
                    return llm.o.get("grades", DocGrades(relevant_ids=[1, 2]))
                if schema is GroundingCheck:
                    return llm.grounding.pop(0) if len(llm.grounding) > 1 else llm.grounding[0]
                if schema is Checklist:
                    return Checklist(titre="Passeport", organisme="Police", documents_requis=["CIN"],
                                     etapes=["Déposer"], frais="Non précisé dans les sources",
                                     delai="Non précisé dans les sources", sources=["Passeport"])
                raise AssertionError(schema)

        return _Runner()

    def invoke(self, messages):
        self.calls.append("generate")
        return AIMessage(content=self.answers.pop(0) if len(self.answers) > 1 else self.answers[0])


@pytest.fixture(scope="session")
def chunks():
    return split_documents(load_documents(settings.docs_dir))


@pytest.fixture(scope="session")
def retriever(chunks):
    return HybridRetriever(chunks)  # BM25 seul (pas d'embeddings en test)
