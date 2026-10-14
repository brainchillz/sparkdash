"""Auto-recovery for whatever engine owns the serving port.

Guards a sparkrun recipe or an engine stack (config.STACKS) the same way:

  1. containers   every expected container is "Up" (a rebooted node loses its
                  container while the other ranks keep answering /health)
  2. /health      answers 200 once the engine is past the startup grace
  3. canary       the periodic real completion (collectors.run_canary) passed

A failure must repeat WATCHDOG_FAIL_THRESHOLD checks in a row, each judged
on fresh evidence: the canary runs on its own (slower) timer, so a check that
would re-count a failed sample it already counted re-runs the canary first
instead of reading the same reply twice. Then it
alerts through the configured webhook, and — if every node is online, no
operation is in progress, and no automatic restart happened within the
cooldown — restarts the engine through the same controller the admin UI
uses, so the restart shows up in the operation panel with streamed output.

Nothing is guarded when nothing is there: a deliberate stop removes the
containers (recipes via sparkrun, stacks via their script), so no marker file
or "desired state" is needed. The check loop keeps a small state dict that the
admin UI shows (last check, failures, last restart).

Replaces the standalone sparkrun-watchdog timer (deploy/sparkrun-watchdog)
and the per-stack cron watchdogs it was modelled on.
"""

from __future__ import annotations

import asyncio
import time

from . import alerts, config, ops, store

state: dict = {
    "last_check": None, "last_status": "not run yet", "failures": 0,
    "last_reason": None, "last_alert": None, "last_restart": None,
    "restarts": 0, "guarding": None, "canary_ts": None,
}


# -- configuration -----------------------------------------------------------

def enabled() -> bool:
    if config.WATCHDOG_LOCKED:
        return config.WATCHDOG_ENABLED_DEFAULT
    v = store.get_setting("watchdog.enabled")
    return config.WATCHDOG_ENABLED_DEFAULT if v is None else v == "1"


def set_enabled(on: bool) -> None:
    store.set_setting("watchdog.enabled", "1" if on else "0")


def status() -> dict:
    return {
        "enabled": enabled(), "locked": config.WATCHDOG_LOCKED,
        "auto_restart": config.WATCHDOG_AUTO_RESTART,
        "interval": config.WATCHDOG_INTERVAL,
        "fail_threshold": config.WATCHDOG_FAIL_THRESHOLD,
        "startup_grace_min": config.WATCHDOG_GRACE_MIN,
        "restart_cooldown_min": config.WATCHDOG_COOLDOWN_MIN,
        "state": dict(state),
    }


# -- checks ------------------------------------------------------------------

def _evaluate(snap: dict) -> str | None:
    """None when the owning engine is serving; else why it is not."""
    eng = snap["engine"]
    if eng["owner"] == "stack" and eng["up"] < eng["expected"]:
        down = [f"{c['name']}@{c['node']}" for c in eng["containers"] if not c["up"]]
        return (f"{eng['up']}/{eng['expected']} containers up"
                + (f"; down: {', '.join(down)}" if down else "")
                + (f"; missing on {eng['expected'] - len(eng['containers'])} node(s)"
                   if len(eng["containers"]) < eng["expected"] else ""))
    if eng["owner"] == "sparkrun" and eng["up"] < eng["expected"]:
        return f"{eng['up']}/{eng['expected']} containers up"
    vllm = snap.get("vllm") or {}
    if not vllm.get("healthy"):
        age = eng.get("youngest_min")
        if age is None:
            age = _recipe_age_min(snap)
        if age is not None and age < config.WATCHDOG_GRACE_MIN:
            return None   # still starting
        return ("/health not answering" if not vllm.get("reachable")
                else "/health failing") + (f" after {age:.0f} min up" if age is not None else "")
    canary = snap.get("canary") or {}
    if (canary and canary.get("ts")
            and time.time() - canary["ts"] < max(2 * config.CANARY_INTERVAL, 600)
            and canary.get("error") != "no model loaded"
            and not canary.get("ok")):
        return ("model canary failed: " + str(canary.get("error") or "incoherent reply")
                + (f"; reply: {canary['reply'][:80]!r}" if canary.get("reply") else ""))
    return None


def _recipe_age_min(snap: dict) -> float | None:
    from . import stacks
    rec = snap.get("recipe") or {}
    ages = [a for a in (stacks.up_minutes(c.get("status", ""))
                        for c in rec.get("containers") or []) if a is not None]
    return min(ages) if ages else None


async def _restart(snap: dict) -> None:
    from . import recipe_ops, stacks
    eng = snap["engine"]
    if eng["owner"] == "stack":
        st = stacks.get(eng["name"])
        if not st:
            raise RuntimeError(f"stack {eng['name']} is not configured")
        stacks.controller.restart(st)
    else:
        recipe_ops.controller.restart()
    await ops.state.wait()
    op = ops.state.status()
    if op.get("phase") != "done":
        raise RuntimeError(op.get("error") or op.get("message") or "restart failed")


# -- loop --------------------------------------------------------------------

async def check(hub) -> None:
    """One watchdog pass (also callable from the admin UI)."""
    snap = hub.snapshot()
    eng = snap.get("engine") or {}
    state["last_check"] = time.time()
    if eng.get("owner") not in ("sparkrun", "stack"):
        state.update(failures=0, last_reason=None, guarding=None,
                     last_status="nothing to guard" if eng.get("owner") != "foreign"
                     else "unmanaged process on the serving port; not guarded")
        return
    state["guarding"] = eng.get("label")
    if ops.state.busy():
        state["last_status"] = f"operation in progress ({ops.state.status().get('action')}); skipped"
        return
    reason = _evaluate(snap)
    if reason is not None and reason.startswith("model canary failed"):
        # One failed canary sample counts once. If this is the sample the
        # previous check already counted, get a fresh one before judging.
        ts = (snap.get("canary") or {}).get("ts")
        if ts is not None and ts == state.get("canary_ts"):
            await hub.run_canary()
            snap = hub.snapshot()
            reason = _evaluate(snap)
        state["canary_ts"] = (snap.get("canary") or {}).get("ts")
    if reason is None:
        if state["failures"] >= config.WATCHDOG_FAIL_THRESHOLD:
            await alerts.send("recovered", f"{eng['label']} is serving again.")
        state.update(failures=0, last_reason=None, last_alert=None,
                     last_status=f"ok: {eng['label']} {eng['up']}/{eng['expected']} containers, healthy")
        return
    state["failures"] += 1
    state["last_reason"] = reason
    state["last_status"] = f"FAIL {state['failures']}/{config.WATCHDOG_FAIL_THRESHOLD}: {reason}"
    print(f"[sparkdash] watchdog: {state['last_status']}", flush=True)
    if state["failures"] < config.WATCHDOG_FAIL_THRESHOLD:
        return
    if state["last_alert"] != reason:
        await alerts.send("engine dead", f"{eng['label']}: {reason}")
        state["last_alert"] = reason
    offline = [n["name"] for n in snap.get("nodes", []) if not n.get("online")]
    if offline:
        msg = f"waiting: node(s) offline: {', '.join(offline)}"
        if state["last_status"] != msg:
            await alerts.send("waiting for node", f"{eng['label']}: {msg}")
        state["last_status"] = msg
        return
    if not config.WATCHDOG_AUTO_RESTART:
        state["last_status"] = f"dead ({reason}); auto-restart is off"
        return
    since = time.time() - (state["last_restart"] or 0)
    cooldown = config.WATCHDOG_COOLDOWN_MIN * 60
    if since < cooldown:
        state["last_status"] = (f"dead ({reason}); last automatic restart "
                                f"{since / 60:.0f} min ago, cooldown {cooldown / 60:.0f} min")
        return
    await alerts.send("restarting", f"{eng['label']}: {reason}. Restarting it now.")
    state.update(last_status=f"restarting {eng['label']}: {reason}",
                 last_restart=time.time(), restarts=state["restarts"] + 1)
    print(f"[sparkdash] watchdog: restarting {eng['label']}: {reason}", flush=True)
    try:
        await _restart(snap)
    except Exception as exc:
        state.update(last_status=f"restart failed: {exc}")
        await alerts.send("restart failed", f"{eng['label']}: {exc}")
        return
    state.update(failures=0, last_reason=None, last_alert=None,
                 last_status=f"restarted {eng['label']} (healthy)")
    await alerts.send("restarted", f"{eng['label']} is back and healthy.")


async def watch(hub) -> None:
    while True:
        await asyncio.sleep(config.WATCHDOG_INTERVAL)
        if not enabled():
            state["last_status"] = "disabled"
            continue
        try:
            await check(hub)
        except Exception as exc:
            state["last_status"] = f"check error: {type(exc).__name__}: {exc}"
            print(f"[sparkdash] watchdog: {state['last_status']}", flush=True)
