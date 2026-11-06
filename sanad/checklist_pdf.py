"""Export de la fiche démarche (Checklist) en PDF, compatible français et arabe."""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

from .schemas import Checklist
from .text import arabic_ratio

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font() -> str:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            try:
                pdfmetrics.registerFont(TTFont("SanadFont", p))
                return "SanadFont"
            except Exception:
                continue
    return "Helvetica"  # repli : l'arabe ne s'affichera pas sans police Unicode


def _shape(text: str) -> str:
    """Mise en forme de l'arabe (liaison des lettres + sens droite-à-gauche)."""
    if arabic_ratio(text) < 0.3:
        return escape(text)
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return escape(get_display(arabic_reshaper.reshape(text)))
    except ImportError:
        return escape(text)


def render_checklist_pdf(c: Checklist) -> bytes:
    font = _font()
    rtl = arabic_ratio(c.titre + " ".join(c.etapes)) > 0.3
    align = TA_RIGHT if rtl else TA_LEFT
    title = ParagraphStyle("t", fontName=font, fontSize=17, leading=22, textColor=colors.HexColor("#0f3d3e"), alignment=align)
    h = ParagraphStyle("h", fontName=font, fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=4,
                       textColor=colors.HexColor("#14532d"), alignment=align)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=15, alignment=align)
    small = ParagraphStyle("s", fontName=font, fontSize=8.5, leading=11, textColor=colors.grey, alignment=align)

    L = (lambda fr, ar: ar) if rtl else (lambda fr, ar: fr)

    def bullets(items: list[str], numbered: bool = False):
        return ListFlowable([ListItem(Paragraph(_shape(i), body)) for i in items] or [ListItem(Paragraph("—", body))],
                            bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                            leftIndent=14, bulletFontName=font, bulletFontSize=9)

    story = [
        Paragraph(_shape(c.titre), title), Spacer(1, 4),
        Paragraph(_shape(f"{L('Organisme', 'الجهة')} : {c.organisme}"), body),
        Paragraph(_shape(L("☐ Documents à fournir", "الوثائق المطلوبة")), h), bullets(c.documents_requis),
        Paragraph(_shape(L("Étapes", "المراحل")), h), bullets(c.etapes, numbered=True),
        Paragraph(_shape(L("Frais", "المعاليم")), h), Paragraph(_shape(c.frais), body),
        Paragraph(_shape(L("Délai", "الآجال")), h), Paragraph(_shape(c.delai), body),
    ]
    if c.remarques:
        story += [Paragraph(_shape(L("Remarques", "ملاحظات")), h), bullets(c.remarques)]
    story += [Spacer(1, 14),
              Paragraph(_shape(L("Sources", "المصادر") + " : " + ", ".join(c.sources)), small),
              Paragraph(escape(f"Généré par SANAD le {date.today():%d/%m/%Y} — informations à confirmer auprès de "
                               "l'administration concernée."), small)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title=c.titre, author="SANAD").build(story)
    return buf.getvalue()
