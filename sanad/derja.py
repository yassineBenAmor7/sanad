"""Détection de langue et normalisation du dialecte tunisien (derja).

Deux étapes :
1. Détection déterministe (script arabe / marqueurs derja / arabizi avec chiffres 3, 7, 9).
2. Glossaire derja -> français injecté comme indice dans le prompt de réécriture du LLM.
   Le glossaire seul produit déjà une requête exploitable si le LLM est indisponible.
"""
from __future__ import annotations

import re

from .text import arabic_ratio, strip_accents

# Glossaire derja (arabizi latin) -> français
LEXICON_LATIN: dict[str, str] = {
    "kifech": "comment", "kifach": "comment", "kifeh": "comment",
    "chnowa": "quoi", "chnoua": "quoi", "chnia": "quels", "chneya": "quels", "chnu": "quoi",
    "win": "où", "fin": "où", "wa9tech": "quand", "waqtech": "quand",
    "9adech": "combien", "qadech": "combien", "9addech": "combien",
    "3lech": "pourquoi", "alech": "pourquoi",
    "nekhou": "obtenir", "nakhou": "obtenir", "nekhth": "obtenir", "nkhar9": "obtenir", "nekhrej": "obtenir",
    "na3mel": "faire", "n3mel": "faire", "ndir": "faire", "na3mlou": "faire",
    "nbadel": "changer", "nbaddel": "changer", "njadded": "renouveler", "njaded": "renouveler",
    "lazem": "nécessaire", "lazmin": "nécessaires", "lezem": "nécessaire",
    "wra9": "documents", "wraq": "documents", "awra9": "documents", "lwra9": "documents", "papiers": "documents",
    "bech": "pour", "besh": "pour", "bch": "pour",
    "chirka": "entreprise", "charika": "entreprise",
    "karhba": "voiture", "karhb": "voiture", "lkarhba": "voiture",
    "carta": "carte d'identité", "lcarta": "carte d'identité", "cin": "carte d'identité",
    "dhayya3t": "j'ai perdu", "dhaya3t": "j'ai perdu", "dhaa3et": "perdue", "tsorket": "volée",
    "chrit": "j'ai acheté", "nechri": "acheter", "nbi3": "vendre",
    "baladiya": "municipalité", "lbaladiya": "municipalité",
    "boulis": "police", "markez": "poste de police",
    "khidma": "travail", "nekhdem": "travailler",
    "yekhou": "prend", "wa9t": "temps", "wqt": "temps",
    "mte3i": "mon", "mta3i": "mon", "mte3": "de", "mta3": "de",
    "najem": "je peux", "nejem": "je peux", "najjem": "je peux",
    "barcha": "beaucoup", "taw": "maintenant", "ena": "je", "inti": "tu",
    "el": "le", "l": "le", "fil": "dans le",
    "tabe3": "timbre", "taba3": "timbre",
    "bultan": "bulletin", "boultan": "bulletin",
}

# Marqueurs derja en écriture arabe (distinguent la derja de l'arabe standard)
DERJA_ARABIC_MARKERS = {
    "كيفاش", "شنوة", "شنوا", "شنية", "شكون", "وقتاش", "قداش", "علاش", "باش", "نجم", "نحب",
    "توا", "برشا", "ياسر", "متاع", "متاعي", "فما", "مافماش", "نخرج", "نعمل", "نبدل", "البولتان",
    "الكرهبة", "كرهبة", "الكارطة", "ضيعت", "وين", "هاذي", "هاذا",
}
DERJA_ARABIC_LEXICON = {
    "كيفاش": "comment", "شنوة": "quoi", "شنوا": "quoi", "شنية": "quels", "قداش": "combien",
    "وين": "où", "باش": "pour", "نجم": "je peux", "نخرج": "obtenir", "نعمل": "faire",
    "نبدل": "changer", "البولتان": "bulletin", "الكرهبة": "voiture", "كرهبة": "voiture",
    "الكارطة": "carte d'identité", "ضيعت": "j'ai perdu", "متاعي": "mon", "عدد": "numéro",
}

_WORD = re.compile(r"[\w؀-ۿ']+", re.UNICODE)
_ARABIZI = re.compile(r"^(?=.*[a-z])(?=.*[2379])[a-z2379]+$")  # ex. 9adech, na3mel, wra9
# Mots français fréquents : s'ils dominent, la phrase est du français
_FRENCH_HINTS = {
    "le", "la", "les", "des", "est", "pour", "comment", "quels", "quelles", "documents", "faut",
    "je", "mon", "ma", "une", "un", "du", "de", "que", "qui", "où", "obtenir", "demande",
}


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def detect_language(text: str) -> str:
    """Retourne 'ar', 'derja' ou 'fr'."""
    words = _words(text)
    if not words:
        return "fr"
    if arabic_ratio(text) > 0.5:
        return "derja" if any(w in DERJA_ARABIC_MARKERS for w in words) else "ar"
    plain = [strip_accents(w) for w in words]
    derja_hits = sum(1 for w in plain if w in LEXICON_LATIN and w not in {"el", "l", "cin", "fin", "papiers"})
    arabizi_hits = sum(1 for w in plain if _ARABIZI.match(w))
    french_hits = sum(1 for w in plain if w in _FRENCH_HINTS)
    score = derja_hits + arabizi_hits
    if score >= 2 or (score >= 1 and score >= french_hits):
        return "derja"
    return "fr"


def gloss(text: str) -> str:
    """Traduction mot-à-mot derja -> français (indice pour le LLM et repli sans LLM)."""
    out = []
    for raw in _WORD.findall(text):
        w = raw.lower()
        out.append(LEXICON_LATIN.get(strip_accents(w)) or DERJA_ARABIC_LEXICON.get(w) or raw)
    return " ".join(out)
