"""Recherche hybride : BM25 (lexical) + FAISS (sémantique) fusionnés par Reciprocal Rank Fusion.

- BM25 capte les termes exacts (« bulletin n°3 », « SUARL », « 2018-52 »).
- Les embeddings multilingues (bge-m3) captent le sens, y compris entre arabe et français.
- RRF combine les deux classements sans avoir à calibrer leurs scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import settings
from .text import tokenize


@dataclass
class Hit:
    doc: Document
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion : score(d) = Σ 1 / (k + rang)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Document], vectorstore=None, dense_weight: float = 1.0):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.dense_weight = dense_weight
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    # ---------- construction ----------
    @classmethod
    def load(cls, index_dir: Path | None = None, embeddings=None, use_dense: bool = True) -> "HybridRetriever":
        index_dir = index_dir or settings.index_dir
        path = index_dir / "chunks.json"
        if not path.exists():
            raise FileNotFoundError(f"Index introuvable ({path}). Lancez d'abord : python -m sanad.ingest")
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Document(page_content=r["text"], metadata=r["metadata"]) for r in raw]
        vs = None
        if use_dense:
            from langchain_community.vectorstores import FAISS

            from .llm import get_embeddings

            # Index produit localement par sanad.ingest (fichier de confiance)
            vs = FAISS.load_local(str(index_dir), embeddings or get_embeddings(),
                                  allow_dangerous_deserialization=True)
        return cls(chunks, vs)

    @classmethod
    def from_documents(cls, chunks: list[Document], embeddings=None) -> "HybridRetriever":
        vs = None
        if embeddings is not None:
            from langchain_community.vectorstores import FAISS

            vs = FAISS.from_documents(chunks, embeddings)
        return cls(chunks, vs)

    # ---------- recherche ----------
    def bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]

    def dense_ranking(self, query: str, n: int) -> list[int]:
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=n)
        return [d.metadata["chunk_id"] for d in results]

    def search(self, query: str, k: int | None = None, extra_queries: list[str] | None = None) -> list[Hit]:
        """Recherche hybride ; `extra_queries` permet d'ajouter la question originale (ex. en derja)."""
        k = k or settings.top_k
        n = max(k * 3, 15)
        rankings, bm25_pos, dense_pos = [], {}, {}
        for q in [query, *(extra_queries or [])]:
            b = self.bm25_ranking(q, n)
            d = self.dense_ranking(q, n)
            rankings += [b] + ([d] * int(self.dense_weight) if d else [])
            for r, i in enumerate(b, 1):
                bm25_pos.setdefault(i, r)
            for r, i in enumerate(d, 1):
                dense_pos.setdefault(i, r)
        fused = rrf(rankings)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], bm25_pos.get(i), dense_pos.get(i)) for i in top]
