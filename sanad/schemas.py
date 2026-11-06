"""Schémas Pydantic : sorties structurées du LLM et contrats de l'API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Sorties structurées du LLM ----------
class DomainRoute(BaseModel):
    """Classification de la question."""

    in_domain: bool = Field(description="True si la question concerne une démarche administrative, juridique ou un service public en Tunisie")
    intent: Literal["procedure", "information", "other"] = Field(description="procedure = l'utilisateur veut accomplir une démarche")
    reason: str = Field(description="Justification courte")


class RewrittenQuery(BaseModel):
    query: str = Field(description="Question reformulée en français standard, précise et autonome")


class DocGrades(BaseModel):
    relevant_ids: list[int] = Field(description="Numéros des extraits réellement utiles pour répondre")


class GroundingCheck(BaseModel):
    """Juge LLM : la réponse est-elle entièrement appuyée par les sources ?"""

    grounded: bool
    score: float = Field(ge=0, le=1, description="Part des affirmations appuyées par les sources (0 à 1)")
    unsupported_claims: list[str] = Field(default_factory=list)


class Checklist(BaseModel):
    """Fiche actionnable d'une démarche."""

    titre: str
    organisme: str = Field(description="Administration / guichet compétent")
    documents_requis: list[str]
    etapes: list[str]
    frais: str = Field(description="Frais indiqués par les sources, ou 'Non précisé dans les sources'")
    delai: str = Field(description="Délai indiqué par les sources, ou 'Non précisé dans les sources'")
    remarques: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


# ---------- Contrats de l'API ----------
class Source(BaseModel):
    id: int
    title: str
    section: str = ""
    source: str
    snippet: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list)
    checklist: bool = False


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "not_found", "out_of_domain", "blocked"]
    language: str
    normalized_query: str
    sources: list[Source] = Field(default_factory=list)
    grounding_score: float | None = None
    checklist: Checklist | None = None
    trace: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
