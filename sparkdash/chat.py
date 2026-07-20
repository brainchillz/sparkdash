"""Chat proxy to the running vLLM model.

The dashboard is HTTPS but vLLM serves plain HTTP on :8000, so the browser
can't call it directly (mixed content). This streams `/v1/chat/completions`
through the backend as newline-delimited JSON events — `{"r": …}` for a
reasoning/thinking delta and `{"c": …}` for an answer delta — so the UI can
show a reasoning model's thinking separately from its final answer.
"""

from __future__ import annotations

import json

import httpx

from . import config


async def current_model_id() -> str | None:
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{config.VLLM_BASE}/v1/models")
            data = r.json().get("data", [])
            return data[0]["id"] if data else None
    except Exception:
        return None


async def stream_chat(messages: list[dict], temperature: float, max_tokens: int):
    """Yield content deltas from vLLM's streaming chat completion."""
    model_id = await current_model_id()
    if not model_id:
        yield json.dumps({"c": "[no model loaded]"}) + "\n"
        return
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{config.VLLM_BASE}/v1/chat/completions",
                                 json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                yield json.dumps({"c": f"[vLLM error {resp.status_code}] {body}"}) + "\n"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (ValueError, KeyError, IndexError):
                    continue
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    yield json.dumps({"r": reasoning}) + "\n"
                content = delta.get("content")
                if content:
                    yield json.dumps({"c": content}) + "\n"
