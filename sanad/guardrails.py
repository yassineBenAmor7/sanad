"""Garde-fous : injection de prompt, longueur, masquage des données personnelles, sorties."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .text import normalize_arabic, strip_accents

INJECTION_PATTERNS = [
    # anglais
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |the )?(previous|prior|above) ",
    r"\byou are now\b", r"\bact as\b.*\b(no|without) (rules|restrictions|limits)",
    r"\b(jailbreak|dan mode|developer mode)\b", r"\bdo anything now\b", r"\byou are dan\b", r"\bnow dan\b",
    r"(reveal|show|print|repeat|display) (me )?(your|the) (system )?(prompt|instructions)",
    r"system prompt",
    # français
    r"ignore[rz]? (toutes |tes |les |vos )?(les )?(instructions|regles|consignes)( precedentes)?",
    r"oublie[rz]? (toutes |tes |les |vos )?(instructions|regles|consignes)",
    r"tu es (maintenant|desormais) ",
    r"(affiche|montre|revele|repete|donne)[- ]?(moi )?(ton|le|tes|les) (prompt|instructions|consignes)",
    r"prompt systeme", r"sans (aucune )?(restriction|regle|limite)s?",
    # arabe
    r"تجاهل (كل |جميع )?(التعليمات|الاوامر|القواعد)",
    r"انس(ى)? (كل |جميع )?(التعليمات|القواعد)",
    r"(اكتب|اعرض|اظهر) (لي )?(التعليمات|تعليماتك|البرومبت)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Demandes clairement illicites liées aux documents officiels
ILLICIT_PATTERNS = [
    r"\b(falsifi|contrefa|faux (papiers|documents|passeport|cin|diplome))",
    r"\b(forge|fake) (id|passport|document)",
    r"تزوير",
]
_ILLICIT_RE = [re.compile(p, re.IGNORECASE) for p in ILLICIT_PATTERNS]

_PII = [
    (re.compile(r"\b\d{8}\b"), "[CIN]"),                                   # numéro CIN (8 chiffres)
    (re.compile(r"(?:\+216[\s.-]?)?\b[2-9]\d(?:[\s.-]?\d{3}){2}\b"), "[TEL]"),  # téléphone tunisien
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def _canon(text: str) -> str:
    return normalize_arabic(strip_accents(text.lower()))


def check_input(text: str, max_chars: int = 1000) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty")
    if len(text) > max_chars:
        return GuardResult(False, "too_long")
    canon = _canon(text)
    if any(r.search(canon) for r in _INJECTION_RE):
        return GuardResult(False, "prompt_injection")
    if any(r.search(canon) for r in _ILLICIT_RE):
        return GuardResult(False, "illicit")
    return GuardResult(True)


def mask_pii(text: str) -> str:
    """Masque CIN / téléphone / email avant tout envoi au LLM ou dans les logs."""
    for rx, repl in _PII:
        text = rx.sub(repl, text)
    return text


_CITATION = re.compile(r"\[(\d+)\]")


def cited_ids(answer: str) -> set[int]:
    return {int(m) for m in _CITATION.findall(answer)}


def strip_invalid_citations(answer: str, valid: set[int]) -> str:
    """Supprime les citations [n] qui ne correspondent à aucune source fournie."""
    return _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)


REFUSALS = {
    "prompt_injection": {
        "fr": "Je ne peux pas suivre cette demande. Je réponds uniquement aux questions sur les démarches administratives en Tunisie.",
        "ar": "لا يمكنني تنفيذ هذا الطلب. أجيب فقط عن الأسئلة المتعلقة بالإجراءات الإدارية في تونس.",
    },
    "illicit": {
        "fr": "Je ne peux pas aider à falsifier ou contourner des documents officiels. Je peux en revanche expliquer la démarche légale pour les obtenir.",
        "ar": "لا يمكنني المساعدة في تزوير الوثائق الرسمية، لكن يمكنني شرح الإجراء القانوني للحصول عليها.",
    },
    "out_of_domain": {
        "fr": "Cette question sort de mon domaine. Je suis spécialisé dans les démarches administratives et juridiques en Tunisie (CIN, passeport, état civil, entreprise, CNSS…).",
        "ar": "هذا السؤال خارج مجال اختصاصي. أنا مختص في الإجراءات الإدارية والقانونية في تونس.",
    },
    "not_found": {
        "fr": "Je n'ai pas trouvé cette information dans mes sources. Je préfère ne pas inventer : renseignez-vous auprès de l'administration concernée.",
        "ar": "لم أجد هذه المعلومة في مصادري. أفضّل عدم التخمين: يرجى الاستفسار لدى الإدارة المعنية.",
    },
    "too_long": {
        "fr": "Votre question est trop longue. Merci de la reformuler plus brièvement.",
        "ar": "سؤالك طويل جدًا. يرجى إعادة صياغته باختصار.",
    },
    "empty": {"fr": "Posez-moi une question.", "ar": "اطرح سؤالك."},
}


def refusal(reason: str, language: str) -> str:
    msgs = REFUSALS.get(reason, REFUSALS["out_of_domain"])
    return msgs["ar"] if language == "ar" else msgs["fr"]
