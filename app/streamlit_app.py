"""Interface SANAD — streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanad.checklist_pdf import render_checklist_pdf  # noqa: E402
from sanad.config import settings  # noqa: E402
from sanad.graph import load_agent  # noqa: E402
from sanad.schemas import Checklist  # noqa: E402

st.set_page_config(page_title="SANAD — Démarches tunisiennes", page_icon="🇹🇳", layout="wide")

EXAMPLES = [
    "Quels documents pour une première carte d'identité ?",
    "kifech nekhou passeport? chnowa el wra9 elli lazmin?",
    "ما هي الوثائق المطلوبة لاستخراج جواز السفر؟",
    "bech na3mel chirka suarl, chnia el etapes?",
    "J'ai acheté une voiture d'occasion, comment changer la carte grise ?",
]
STATUS_BADGE = {"answered": "🟢 Réponse sourcée", "not_found": "🟠 Information introuvable",
                "out_of_domain": "🔵 Hors domaine", "blocked": "🔴 Bloqué par les garde-fous"}


@st.cache_resource(show_spinner="Chargement de l'index et du modèle…")
def agent():
    return load_agent()


def render_result(res: dict, key: str) -> None:
    cols = st.columns(4)
    cols[0].caption(STATUS_BADGE.get(res["status"], res["status"]))
    cols[1].caption(f"Langue : **{res['language']}**")
    if res.get("grounding_score") is not None:
        cols[2].caption(f"Fiabilité (vérif. faits) : **{res['grounding_score']:.0%}**")
    cols[3].caption(f"⏱️ {res['latency_s']} s")
    if res["sources"]:
        with st.expander(f"📚 Sources ({len(res['sources'])})"):
            for s in res["sources"]:
                st.markdown(f"**[{s['id']}] {s['title']}** — _{s['section']}_  \n`{s['source']}`")
                st.caption(s["snippet"])
    if st.session_state.show_trace and res.get("trace"):
        with st.expander("🧠 Raisonnement de l'agent (LangGraph)"):
            st.markdown("\n".join(f"{i}. {t}" for i, t in enumerate(res["trace"], 1)))
            if res["normalized_query"]:
                st.caption(f"Requête normalisée : {res['normalized_query']}")
    if res.get("checklist"):
        c = Checklist(**res["checklist"])
        with st.expander("📋 Fiche démarche", expanded=True):
            st.markdown(f"**{c.titre}** — {c.organisme}")
            st.markdown("**Documents :**\n" + "\n".join(f"- [ ] {d}" for d in c.documents_requis))
            st.markdown("**Étapes :**\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(c.etapes, 1)))
            st.markdown(f"**Frais :** {c.frais}  \n**Délai :** {c.delai}")
            st.download_button("⬇️ Télécharger la fiche PDF", render_checklist_pdf(c),
                               file_name="sanad_fiche.pdf", mime="application/pdf", key=f"pdf_{key}")


# ---------- barre latérale ----------
with st.sidebar:
    st.title("🇹🇳 SANAD")
    st.caption("Assistant IA agentique — démarches administratives tunisiennes. Français · العربية · Derja")
    st.session_state.setdefault("show_trace", True)
    want_checklist = st.toggle("Générer une fiche démarche (PDF)", value=True)
    st.session_state.show_trace = st.toggle("Afficher le raisonnement", value=st.session_state.show_trace)
    st.divider()
    st.markdown("**Exemples**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    manifest = settings.index_dir / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        st.caption(f"Index : {m['documents']} documents · {m['chunks']} extraits")
    st.caption(f"LLM : `{settings.llm_model}` · Embeddings : `{settings.embed_model}` — 100 % local (Ollama)")
    if st.button("🗑️ Nouvelle conversation"):
        st.session_state.messages = []
    st.caption("⚠️ Informations indicatives : vérifiez toujours auprès de l'administration.")

# ---------- conversation ----------
st.session_state.setdefault("messages", [])
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result"):
            render_result(m["result"], str(i))

prompt = st.chat_input("Posez votre question (français, عربي, derja)…") or st.session_state.pop("pending", None)
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("SANAD analyse votre question…"):
                res = agent().ask(prompt, history, checklist=want_checklist).model_dump()
        except FileNotFoundError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"Le moteur LLM est indisponible ({type(e).__name__}). Ollama est-il lancé ?"); st.stop()
        st.markdown(res["answer"])
        render_result(res, str(len(st.session_state.messages)))
    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "result": res})
