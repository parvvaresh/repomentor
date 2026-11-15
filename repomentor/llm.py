"""Calls to the local Ollama server, whole or streamed."""

import json
from collections.abc import Iterator

import requests

from . import config


class ModelError(RuntimeError):
    pass


def _payload(messages, temperature, context_length, model, stream) -> dict:
    options: dict = {"temperature": temperature}

    # Ollama's default context window is small; prompts that include
    # several whole sources need a larger one
    if context_length is not None:
        options["num_ctx"] = context_length

    return {
        "model": model or config.MODEL_NAME,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": options,
    }


def _post(payload: dict, timeout: int, stream: bool) -> requests.Response:
    url = f"{config.OLLAMA_URL}/api/chat"

    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=stream)
    except requests.ConnectionError as error:
        raise ModelError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. "
            "Make sure Ollama is running, or use retrieval-only mode."
        ) from error

    if response.status_code == 404:
        raise ModelError(
            f"Ollama does not have the model '{payload['model']}'. "
            f"Run: ollama pull {payload['model']}"
        )

    if not response.ok:
        raise ModelError(f"Ollama returned {response.status_code}: {response.text[:300]}")

    return response


def chat(
    messages: list[dict],
    temperature: float = 0,
    context_length: int | None = None,
    timeout: int = 180,
    model: str | None = None,
) -> str:
    payload = _payload(messages, temperature, context_length, model, stream=False)
    return _post(payload, timeout, stream=False).json()["message"]["content"].strip()


def chat_stream(
    messages: list[dict],
    temperature: float = 0,
    context_length: int | None = None,
    timeout: int = 600,
    model: str | None = None,
) -> Iterator[str]:
    """Yield the answer in pieces as the model writes it."""

    payload = _payload(messages, temperature, context_length, model, stream=True)

    with _post(payload, timeout, stream=True) as response:
        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)

            if data.get("error"):
                raise ModelError(data["error"])

            piece = data.get("message", {}).get("content", "")

            if piece:
                yield piece

            if data.get("done"):
                return


def available() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False
