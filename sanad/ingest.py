"""Ingestion : chargement des documents, découpage structurel, index FAISS + corpus BM25.

Usage : python -m sanad.ingest [--docs data/docs] [--index index]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import settings

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SKIP_FILES = {"readme.md"}


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name.lower() in SKIP_FILES:
            continue
        suffix = path.suffix.lower()
        rel = str(path.relative_to(docs_dir))
        if suffix in {".md", ".txt"}:
            meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
            meta.setdefault("title", path.stem.replace("_", " ").title())
            docs.append(Document(page_content=body, metadata={**meta, "source": rel, "format": "md"}))
        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(Document(page_content=text, metadata={
                        "title": path.stem.replace("_", " "), "source": rel, "page": i, "format": "pdf"}))
    return docs


def split_documents(docs: list[Document], chunk_size: int | None = None, overlap: int | None = None) -> list[Document]:
    """Découpage en 2 temps : par sections (titres Markdown) puis par taille.

    Chaque chunk reçoit un en-tête contextuel « titre > section » : un extrait isolé
    ("Documents requis…") reste ainsi rattaché à sa démarche, ce qui améliore fortement
    la recherche (contextual chunk headers).
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", "، ", " ", ""])

    chunks: list[Document] = []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content) if doc.metadata.get("format") == "md" \
            else [Document(page_content=doc.page_content)]
        for sec in sections:
            section = " > ".join(sec.metadata[h] for h in ("h2", "h3") if sec.metadata.get(h))
            for piece in size_splitter.split_text(sec.page_content):
                piece = piece.strip()
                if len(piece) < 20:
                    continue
                header = doc.metadata["title"] + (f" > {section}" if section else "")
                meta = {k: v for k, v in doc.metadata.items() if k != "format"}
                meta.update(section=section, header=header)
                chunks.append(Document(page_content=f"{header}\n{piece}", metadata=meta))
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def corpus_fingerprint(chunks: list[Document]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.page_content.encode("utf-8"))
    return h.hexdigest()[:16]


def build_index(docs_dir: Path | None = None, index_dir: Path | None = None, embeddings=None) -> dict:
    from langchain_community.vectorstores import FAISS

    from .llm import get_embeddings

    docs_dir = docs_dir or settings.docs_dir
    index_dir = index_dir or settings.index_dir
    embeddings = embeddings or get_embeddings()
    t0 = time.time()
    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"Aucun document trouvé dans {docs_dir}")
    chunks = split_documents(docs)
    index_dir.mkdir(parents=True, exist_ok=True)
    FAISS.from_documents(chunks, embeddings).save_local(str(index_dir))
    # Corpus BM25 sauvegardé en JSON (pas de pickle : chargement sûr)
    (index_dir / "chunks.json").write_text(json.dumps(
        [{"text": c.page_content, "metadata": c.metadata} for c in chunks], ensure_ascii=False, indent=1),
        encoding="utf-8")
    manifest = {
        "documents": len({d.metadata["source"] for d in docs}),
        "chunks": len(chunks),
        "embed_model": getattr(embeddings, "model", type(embeddings).__name__),
        "fingerprint": corpus_fingerprint(chunks),
        "build_seconds": round(time.time() - t0, 2),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Construit l'index SANAD")
    ap.add_argument("--docs", type=Path, default=settings.docs_dir)
    ap.add_argument("--index", type=Path, default=settings.index_dir)
    args = ap.parse_args()
    print(json.dumps(build_index(args.docs, args.index), indent=2))


if __name__ == "__main__":
    main()
