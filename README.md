# TeachMeFinance — LLM-only Chatbot (No RAG)

A simple financial education chatbot using a **free local API** via [Ollama](https://ollama.com/).  
No cloud keys, just a guardrailed LLM chat loop.

## Quickstart
```bash
# 1) Install & run Ollama:
# macOS: brew install ollama && ollama serve
# Linux: curl -fsSL https://ollama.com/install.sh | sh && ollama serve
# Windows: install app from ollama.com and start it

# 2) Pull a model
ollama pull qwen2.5:7b-instruct

# 3) Create venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4) Chat
python chatbot.py chat

# One-shot
python chatbot.py ask "What is a Roth IRA?"
