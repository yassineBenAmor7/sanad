# Publie le projet sur GitHub (Windows PowerShell). Prérequis : Git + GitHub CLI (winget install GitHub.cli)
# Usage : .\scripts\push_to_github.ps1
gh auth status 2>$null; if ($LASTEXITCODE -ne 0) { gh auth login }
gh repo create sanad --public --source . --remote origin `
  --description "Assistant IA agentique (LangGraph, RAG hybride, Ollama) pour les démarches administratives tunisiennes — FR / AR / Derja" `
  --push
gh repo edit --add-topic llm,rag,langgraph,langchain,ollama,faiss,generative-ai,nlp,arabic,tunisia,fastapi,streamlit
Write-Host "Projet publié : $(gh repo view --json url -q .url)"
