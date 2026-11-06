"""Normalisation de texte multilingue (français, arabe, arabizi) pour la recherche lexicale."""
from __future__ import annotations

import re
import unicodedata

_AR_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat + tatweel
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
_ARABIC_CHAR = re.compile(r"[؀-ۿ]")

STOPWORDS = {
    # français
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "et", "ou", "a", "à", "au", "aux", "en",
    "pour", "par", "sur", "dans", "avec", "est", "sont", "que", "qui", "quoi", "quel", "quelle", "quels",
    "quelles", "je", "j", "tu", "il", "on", "nous", "vous", "mon", "ma", "mes", "ce", "cette", "se", "s",
    "faut", "il", "comment", "ne", "pas", "plus", "y", "c", "qu", "n", "m", "puis", "dois", "doit",
    # arabe
    "في", "من", "على", "إلى", "الى", "عن", "ما", "هي", "هو", "او", "أو", "و", "ب", "ل", "مع", "هذا", "هذه",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_arabic(text: str) -> str:
    text = _AR_DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _ARABIC_CHAR.match(c)) / len(letters)


def tokenize(text: str) -> list[str]:
    """Tokenisation pour BM25 : minuscules, sans accents, arabe normalisé, sans mots vides."""
    text = normalize_arabic(strip_accents(text.lower()))
    tokens = [t for t in _TOKEN.findall(text) if t not in STOPWORDS and len(t) > 1]
    # préfixe arabe "ال" : on indexe aussi la forme sans article
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if t.startswith("ال") and len(t) > 4:
            out.append(t[2:])
    return out
