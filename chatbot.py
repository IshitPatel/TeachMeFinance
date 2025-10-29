#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, time
from typing import List, Dict, Any, Optional
import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(help="TeachMeFinance — LLM-only chatbot (no RAG) using Ollama local API")
console = Console()

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SYSTEM_PROMPT = """You are TeachMeFinance, a *financial education* assistant.
Your goals:
- Explain personal-finance concepts clearly (budgeting, saving, emergency funds, diversification, index funds, retirement accounts, risk tolerance).
- Be concise, structured, and beginner-friendly; use examples or rules of thumb where helpful.
- You are **not** a financial advisor; avoid personalized investment advice or fiduciary recommendations.
- If the user asks for individualized advice, provide general education and suggest consulting a licensed professional for personal recommendations.
- Avoid giving tax/legal advice; you may explain general concepts with disclaimers.
- When math is needed, show steps briefly and state assumptions.
"""

def chat_completion_ollama(model: str, messages: List[Dict[str, str]],
                           temperature: float = 0.4,
                           max_tokens: int = 512,
                           base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """
    Call Ollama /api/chat with the provided messages.
    Assumes Ollama is running locally with `ollama serve`.
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    data = resp.json()
    # Non-streaming returns a single dictionary with 'message'
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    # Some builds may return aggregated content differently
    if "response" in data:
        return str(data["response"]).strip()
    raise RuntimeError("Unexpected Ollama response schema")

def banner(model: str) -> None:
    console.print(Panel.fit(f"[bold magenta]TeachMeFinance[/] — LLM-only (model: {model}) — type 'exit' to quit",
                            border_style="magenta"))

@app.command()
def chat(model: str = typer.Option("qwen2.5:7b-instruct", "--model", "-m", help="Ollama model name"),
         temperature: float = typer.Option(0.4, "--temperature", "-t"),
         max_tokens: int = typer.Option(512, "--max-tokens"),
         ) -> None:
    """
    Interactive REPL chat.
    """
    banner(model)
    history: List[Dict[str,str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        q = Prompt.ask("[bold cyan]You[/]").strip()
        if q.lower() in {"exit", "quit"}:
            break
        history.append({"role":"user","content": q})
        try:
            answer = chat_completion_ollama(model, history, temperature, max_tokens)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            continue
        history.append({"role":"assistant","content": answer})
        console.print(Panel(answer, title="TeachMeFinance", border_style="cyan"))

@app.command()
def ask(question: str = typer.Argument(..., help="Your finance education question"),
        model: str = typer.Option("qwen2.5:7b-instruct", "--model", "-m", help="Ollama model name"),
        temperature: float = typer.Option(0.4, "--temperature", "-t"),
        max_tokens: int = typer.Option(512, "--max-tokens"),
        ) -> None:
    """
    Single-shot answer for a given question; prints the response and exits.
    """
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": question}
    ]
    try:
        answer = chat_completion_ollama(model, messages, temperature, max_tokens)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)
    console.print(answer)

if __name__ == "__main__":
    app()
