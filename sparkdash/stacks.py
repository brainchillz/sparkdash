"""Engine stacks: inference engines launched outside sparkrun.

A stack (see config.STACKS) is one container per node started by an operator
script — the kind of hand-tuned vLLM build that no sparkrun recipe describes.
SparkDash does not try to understand the script; it only needs three things
from it, and they are what `~/glm`-style wrappers already provide:

  start    launch every rank and block until the engine is healthy
           (exit non-zero if it is not), then any warm-up
  stop     tear the containers down on every node
  restart  stop, then start

Detection is by container-name prefix in the per-node probe (collectors.py),
so a stack started by hand on the console shows up on the dashboard exactly
like one started here. Operations run through ops.state, the same one-at-a-
time, streamed-output panel the recipe controller uses.
"""

from __future__ import annotations

import asyncio
import os
import re

import httpx

from . import config, ops

_UP_RE = re.compile(
    r"^Up\s+(?:(\d+)\s+(second|minute|hour|day|week)s?|About a (minute|hour)|Less than a second)",
    re.I)


def get(name: str) -> dict | None:
    for st in config.STACKS:
        if st["name"] == name:
            return st
    return None


def up_minutes(status: str) -> float | None:
    """Minutes a container has been up, from docker's 'Up 9 minutes' text
    (None when the container is not up)."""
    m = _UP_RE.match((status or "").strip())
    if not m:
        return None
    if m.group(3):
        return 1.0 if m.group(3).lower() == "minute" else 60.0
    if m.group(1) is None:
        return 0.0
    n, unit = float(m.group(1)), m.group(2).lower()
    return n * {"second": 1 / 60, "minute": 1, "hour": 60, "day": 1440,
                "week": 10080}[unit]


def describe(stack: dict, probe: dict[str, dict]) -> dict:
    """What the node probe sees of `stack`: its containers across nodes."""
    conts = []
    for n in config.NODES:
        for c in probe.get(n["name"], {}).get("stack_containers", []):
            if c["name"].startswith(stack["prefix"]):
                conts.append({"node": n["name"], "name": c["name"],
                              "status": c["status"],
                              "up": c["status"].startswith("Up")})
    up = [c for c in conts if c["up"]]
    ages = [a for a in (up_minutes(c["status"]) for c in up) if a is not None]
    return {
        "name": stack["name"], "label": stack["label"], "model": stack["model"],
        "present": bool(conts), "containers": conts,
        "expected": stack["expected"], "up": len(up),
        "youngest_min": min(ages) if ages else None,
        "stopped_marker": bool(stack["stopped_marker"]
                               and os.path.exists(stack["stopped_marker"])),
    }


# -- pre-flight --------------------------------------------------------------

async def preflight(stack: dict, snapshot: dict) -> dict:
    """Can this stack be started right now? Cheap checks, no side effects:
    nothing else owns the port, every node is online, the script exists."""
    problems: list[str] = []
    eng = snapshot.get("engine") or {}
    owner = eng.get("owner")
    if owner == "sparkrun":
        problems.append(f"recipe {eng.get('name')} owns the serving port — stop it first")
    elif owner == "stack" and eng.get("name") != stack["name"]:
        problems.append(f"stack {eng.get('label')} owns the serving port — stop it first")
    elif owner == "stack":
        problems.append(f"{stack['label']} is already running")
    elif owner == "foreign":
        problems.append("something not managed here answers the serving port")
    offline = [n["name"] for n in snapshot.get("nodes", []) if not n.get("online")]
    if offline:
        problems.append("node(s) offline: " + ", ".join(offline))
    if not os.access(stack["command"], os.X_OK):
        problems.append(f"command not executable: {stack['command']}")
    return {"ok": not problems, "problems": problems,
            "stack": stack["name"], "label": stack["label"]}


# -- operations --------------------------------------------------------------

class StackController:
    """Drive a stack through its command. One op at a time (ops.state)."""

    def status(self) -> dict:
        return ops.state.status()

    def start(self, stack: dict) -> None:
        ops.state.launch("start", self._run(stack, "start"),
                         engine=f"stack:{stack['name']}", recipe=stack["label"])

    def stop(self, stack: dict) -> None:
        ops.state.launch("stop", self._run(stack, "stop"),
                         engine=f"stack:{stack['name']}", recipe=stack["label"])

    def restart(self, stack: dict) -> None:
        ops.state.launch("restart", self._run(stack, "restart"),
                         engine=f"stack:{stack['name']}", recipe=stack["label"])

    async def _run(self, stack: dict, action: str) -> None:
        verb = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}[action]
        try:
            ops.state.touch(phase=action + "ing" if action != "stop" else "stopping",
                            message=f"{verb} {stack['label']}")
            rc = await ops.state.stream([stack["command"], action])
            if rc != 0:
                ops.state.finish(False, f"{verb.rstrip('ing')} failed",
                                 f"{os.path.basename(stack['command'])} {action} exited {rc}")
                return
            if action == "stop":
                ops.state.finish(True, f"{stack['label']} stopped")
                return
            # The script blocks until healthy; confirm from here anyway.
            if await _healthy():
                ops.state.finish(True, f"{stack['label']} running — engine healthy")
            else:
                ops.state.finish(False, f"{stack['label']} script finished but "
                                 "the engine is not healthy", "health check failed")
        except Exception as exc:
            ops.state.finish(False, f"{verb.rstrip('ing')} failed",
                             f"{type(exc).__name__}: {exc}")


async def _healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{config.VLLM_BASE}/health")
            return r.status_code == 200
    except Exception:
        return False


controller = StackController()
