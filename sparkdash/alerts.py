"""Push alerts on cluster state transitions.

Every incident so far was discovered by a human happening to look at the
page. This watches the merged snapshot and POSTs a short message when the
state actually *changes*: healthy <-> degraded, the loaded model changes, or
a node drops offline.

Configuration tiers, first match wins (mirrors sso.py — the install-time
decision always wins):

  1. an [alerts] table in config.toml           (locked: UI read-only)
  2. the settings table in the admin DB, managed from the admin UI

The watcher re-resolves the configuration every check, so setting or
removing the webhook from the UI takes effect without a restart.

Three body styles:
  * json (default) - {"title": ..., "message": ..., "ts": ...}
  * ntfy           - plain-text body with a Title header, which is what
                     ntfy.sh-compatible servers expect.
  * gchat          - {"text": "*title*\nmessage"} for a Google Chat space's
                     incoming webhook.

A transition must survive two consecutive checks (~20 s) before it alerts, so
a single flapped poll doesn't page anyone.
"""

from __future__ import annotations

import asyncio
import socket
import time

import httpx

from . import config, store

_CHECK_INTERVAL = 10.0
STYLES = ("json", "ntfy", "gchat")


def _tag() -> str:
    # Several installs may share one destination; say who is talking.
    return f"SparkDash {socket.gethostname().split('.')[0]}"


# -- configuration -----------------------------------------------------------

def get_config() -> dict | None:
    """Resolve the active configuration, or None."""
    if config.ALERT_WEBHOOK:
        return {"webhook_url": config.ALERT_WEBHOOK,
                "style": config.ALERT_STYLE if config.ALERT_STYLE in STYLES
                else "json",
                "source": "config"}
    url = store.get_setting("alerts.webhook_url")
    if url:
        style = store.get_setting("alerts.style") or "json"
        return {"webhook_url": url,
                "style": style if style in STYLES else "json",
                "source": "stored"}
    return None


def locked() -> bool:
    """True when config.toml fixes this and the UI must not edit it."""
    return bool(config.ALERT_WEBHOOK)


def enabled() -> bool:
    return get_config() is not None


def save_stored(url: str, style: str) -> None:
    store.set_setting("alerts.webhook_url", url)
    store.set_setting("alerts.style", style)


def clear_stored() -> bool:
    removed = store.delete_setting("alerts.webhook_url")
    store.delete_setting("alerts.style")
    return removed


# -- state diffing -----------------------------------------------------------

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
        out.append((f"{_tag()}: cluster degraded", why))
    elif not prev["healthy"] and cur["healthy"]:
        out.append((f"{_tag()}: cluster healthy",
                    f"model {cur['model'] or '—'} serving"))
    if cur["model"] != prev["model"]:
        out.append((f"{_tag()}: model changed",
                    f"{prev['model'] or '—'} → {cur['model'] or '—'}"))
    newly_off = set(cur["offline"]) - set(prev["offline"])
    if newly_off:
        out.append((f"{_tag()}: node offline", ", ".join(sorted(newly_off))))
    return out


# -- delivery ----------------------------------------------------------------

async def _send(client: httpx.AsyncClient, cfg: dict,
                title: str, message: str) -> httpx.Response:
    if cfg["style"] == "ntfy":
        return await client.post(cfg["webhook_url"], content=message.encode(),
                                 headers={"Title": title})
    if cfg["style"] == "gchat":
        return await client.post(cfg["webhook_url"],
                                 json={"text": f"*{title}*\n{message}"})
    return await client.post(cfg["webhook_url"], json={
        "title": title, "message": message, "ts": time.time()})


async def send_test(cfg: dict | None = None) -> str | None:
    """Fire one test message through `cfg` (default: the active config).
    Returns None on success, or a short error string."""
    cfg = cfg or get_config()
    if not cfg:
        return "Alerts are not configured."
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await _send(client, cfg, f"{_tag()}: test alert",
                            "Webhook delivery from this install is working.")
            r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"Webhook answered HTTP {exc.response.status_code}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


async def watch(hub) -> None:
    """Background task: diff the snapshot state and notify on stable changes."""
    client = httpx.AsyncClient(timeout=10.0)
    confirmed = _state_of(hub.snapshot())   # alerted baseline
    pending: dict | None = None             # candidate awaiting confirmation
    try:
        while True:
            await asyncio.sleep(_CHECK_INTERVAL)
            snap = hub.snapshot()
            cur = _state_of(snap)
            cfg = get_config()          # re-resolved so UI changes apply live
            if cfg is None:
                # Track state silently: turning alerts on later must not
                # replay transitions that happened while they were off.
                confirmed, pending = cur, None
                continue
            if cur == confirmed:
                pending = None
                continue
            if pending != cur:
                pending = cur               # first sighting — wait one round
                continue
            for title, message in _describe(confirmed, cur, snap):
                try:
                    await _send(client, cfg, title, message)
                except Exception as exc:
                    print(f"[sparkdash] alert_send: {type(exc).__name__}: {exc}",
                          flush=True)
            confirmed, pending = cur, None
    finally:
        await client.aclose()
