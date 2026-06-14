# ollama_client.py — configured Ollama connector

import ollama

from config import OLLAMA_HOST, OLLAMA_REQUEST_TIMEOUT_SEC


_CLIENT = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_REQUEST_TIMEOUT_SEC)


def chat(model: str, messages: list, options: dict):
    """Call Ollama chat through the configured host with thinking disabled."""
    return _CLIENT.chat(
        model=model,
        messages=messages,
        options=options,
        think=False,
    )


def generate(model: str, prompt: str, keep_alive: int):
    """Call Ollama generate through the configured host with thinking disabled."""
    return _CLIENT.generate(
        model=model,
        prompt=prompt,
        keep_alive=keep_alive,
        think=False,
    )


def list_models():
    """List models from the configured Ollama host."""
    return _CLIENT.list()
