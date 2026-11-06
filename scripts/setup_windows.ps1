# Installation SANAD sous Windows (PowerShell) — prérequis : Python 3.11+ et Ollama (https://ollama.com)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
ollama pull qwen2.5:7b
ollama pull bge-m3
python -m sanad.ingest
pytest -q
Write-Host "OK. Lancez : streamlit run app/streamlit_app.py"
