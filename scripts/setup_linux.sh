#!/usr/bin/env bash
# Installation SANAD sous Linux / macOS — prérequis : Python 3.11+ et Ollama
set -e
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env || true
ollama pull qwen2.5:7b
ollama pull bge-m3
python -m sanad.ingest
pytest -q
echo "OK. Lancez : streamlit run app/streamlit_app.py"
