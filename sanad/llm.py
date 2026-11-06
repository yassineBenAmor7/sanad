"""Fabrique des modèles Ollama (100 % local : aucune donnée ne quitte la machine)."""
from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature if temperature is None else temperature,
        num_ctx=8192,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
