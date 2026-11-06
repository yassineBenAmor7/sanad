import json
from pathlib import Path

import pytest

from sanad.derja import detect_language, gloss
from sanad.guardrails import check_input, mask_pii, strip_invalid_citations
from sanad.retriever import rrf
from sanad.text import normalize_arabic, tokenize

QUESTIONS = [json.loads(l) for l in (Path(__file__).parent.parent / "data/eval/questions.jsonl").open(encoding="utf-8")]


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["type"] in {"fr", "ar", "derja"}], ids=lambda q: str(q["id"]))
def test_language_detection(q):
    assert detect_language(q["question"]) == q["type"]


def test_gloss_translates_derja():
    g = gloss("kifech nekhou passeport? chnowa el wra9 elli lazmin?")
    assert "comment" in g and "obtenir" in g and "documents" in g


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["type"] == "injection"], ids=lambda q: str(q["id"]))
def test_injections_blocked(q):
    assert not check_input(q["question"]).allowed


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["type"] in {"fr", "ar", "derja", "out_of_domain"}],
                         ids=lambda q: str(q["id"]))
def test_legit_questions_not_blocked(q):
    assert check_input(q["question"]).allowed


def test_illicit_and_length():
    assert check_input("comment faire un faux passeport").reason == "illicit"
    assert check_input("a" * 2000).reason == "too_long"


def test_pii_masking():
    out = mask_pii("CIN 12345678, tel +216 22 333 444, mail x@y.tn")
    assert "12345678" not in out and "333" not in out and "x@y.tn" not in out


def test_strip_invalid_citations():
    assert strip_invalid_citations("A [1] B [7]", {1, 2}) == "A [1] B "


def test_arabic_normalization_and_tokens():
    assert normalize_arabic("أَحْمَد") == "احمد"
    assert "سفر" in " ".join(tokenize("جواز السفر"))


def test_rrf_prefers_consensus():
    s = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(s, key=s.get) in {1, 2} and s[2] > s[3]


def test_chunks_have_context_headers(chunks):
    assert all(c.page_content.startswith(c.metadata["title"]) for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["expected_source"]], ids=lambda q: str(q["id"]))
def test_retrieval_hit_at_6(retriever, q):
    query = gloss(q["question"]) if detect_language(q["question"]) == "derja" else q["question"]
    hits = retriever.search(query, k=6, extra_queries=[q["question"]])
    assert q["expected_source"] in [h.doc.metadata["source"] for h in hits]
