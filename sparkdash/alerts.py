"""Push alerts on cluster state transitions.

Every incident so far was discovered by a human happening to look at the
page. This watches the merged snapshot and POSTs a short message when the
state actually *changes*: healthy <-> degraded, the loaded model changes, or
a node drops offline. Off unless [alerts].webhook_url is set in config.toml.

Two body styles:
  * json (default) - {"title": ..., "message": ..., "ts": ...}
  * ntfy           - plain-text body with a Title header, which is what
                     ntfy.sh-compatible servers expect.

A transition must survive two consecutive checks (~20 s) before it alerts, so
a single flapped poll doesn't page anyone.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from . import config

_CHECK_INTERVAL = 10.0


def _state_of(snap: dict) -> dict:
    return {
        "healthy": bool(snap.get("cluster_healthy")),
        "model": (snap.get("vllm") or {}).get("model"),
        "offline": tuple(sorted(n["name"] for n in snap.get("nodes", [])
                                if not n.get("online"))),
    }


def _describe(prev: dict, cur: dict, snap: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if prev["healthy"] and not cur["healthy"]:
        why = "; ".join(snap.get("health_reasons") or []) or "unknown"
        out.append(("SparkDash: cluster degraded", why))
    elif not prev["healthy"] and cur["healthy"]:
        out.append(("SparkDash: cluster healthy",
                    f"model {cur['model'] or '—'} serving"))
    if cur["model"] != prev["model"]:
        out.append(("SparkDash: model changed",
                    f"{prev['model'] or '—'} → {cur['model'] or '—'}"))
    newly_off = set(cur["offline"]) - set(prev["offline"])
    if newly_off:
        out.append(("SparkDash: node offline", ", ".join(sorted(newly_off))))
    return out


async def _send(client: httpx.AsyncClient, title: str, message: str) -> None:
    if config.ALERT_STYLE == "ntfy":
        await client.post(config.ALERT_WEBHOOK, content=message.encode(),
                          headers={"Title": title})
    else:
        await client.post(config.ALERT_WEBHOOK, json={
            "title": title, "message": message, "ts": time.time()})


async def watch(hub) -> None:
    """Background task: diff the snapshot state and notify on stable changes."""
    if not config.ALERT_WEBHOOK:
        return
    client = httpx.AsyncClient(timeout=10.0)
    confirmed = _state_of(hub.snapshot())   # alerted baseline
    pending: dict | None = None             # candidate awaiting confirmation
    try:
        while True:
            await asyncio.sleep(_CHECK_INTERVAL)
            snap = hub.snapshot()
            cur = _state_of(snap)
            if cur == confirmed:
                pending = None
                continue
            if pending != cur:
                pending = cur               # first sighting — wait one round
                continue
            for title, message in _describe(confirmed, cur, snap):
                try:
                    await _send(client, title, message)
                except Exception as exc:
                    print(f"[sparkdash] alert_send: {type(exc).__name__}: {exc}",
                          flush=True)
            confirmed, pending = cur, None
    finally:
        await client.aclose()
