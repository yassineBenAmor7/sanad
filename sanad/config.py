"""Configuration centralisée (variables d'environnement / fichier .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _path(var: str, default: str) -> Path:
    p = Path(os.getenv(var, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("SANAD_LLM", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: os.getenv("SANAD_EMBED", "bge-m3"))
    temperature: float = field(default_factory=lambda: float(os.getenv("SANAD_TEMPERATURE", "0.1")))
    top_k: int = field(default_factory=lambda: int(os.getenv("SANAD_TOP_K", "6")))
    max_rewrites: int = field(default_factory=lambda: int(os.getenv("SANAD_MAX_REWRITES", "2")))
    max_regenerations: int = 1
    max_question_chars: int = 1000
    chunk_size: int = 700
    chunk_overlap: int = 100
    index_dir: Path = field(default_factory=lambda: _path("SANAD_INDEX_DIR", "index"))
    docs_dir: Path = field(default_factory=lambda: _path("SANAD_DOCS_DIR", "data/docs"))


settings = Settings()
