"""Launch-time failure detection for sparkrun recipes.

`sparkrun run --no-follow` exits 0 as soon as the containers exist, and each
container is `sleep infinity` with vLLM launched by a detached `docker exec`
that writes to /tmp/sparkrun_serve.log inside the container. So when vLLM
dies or hangs during startup the container stays "Up", `sparkrun status`
says running, `docker logs` shows only Ray, and anything that merely polls
/health waits forever. Every "start last hung" episode from 2026-07 on was
this blind spot — the traceback was in the serve log the whole time.

This module reads what the health poll cannot:

  * the serve PID (sparkrun 0.3.6+ writes /tmp/sparkrun_serve.pid) — dead
    process, dead launch;
  * the serve log — fatal patterns (EngineDeadError, CUDA errors, a squatted
    port, argparse failures, NCCL faults) end the wait with the excerpt;
  * log growth — a launch that has written nothing for LAUNCH_STALL_MIN
    while /health still fails is a hang, however young it is. A launch that
    keeps logging (weights, graph capture) is allowed up to LAUNCH_MAX_MIN,
    so a 25-minute cold start is not a timeout.

`watch()` drives one launch and reports (ok, error, excerpt); `save_logs()`
captures every rank's log to DATA_DIR/launch-logs before a stop removes the
containers (and the evidence) — the old watchdog wiped them every time.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import httpx

from . import config

SERVE_LOG = "/tmp/sparkrun_serve.log"
SERVE_PID = "/tmp/sparkrun_serve.pid"
LOG_DIR = str(config.DATA_DIR / "launch-logs")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Seen in this cluster's serve logs right before (or instead of) a start.
# A match opens a short confirmation window rather than failing outright: if
# the process exits or the log goes silent in that window, it was fatal; if
# vLLM keeps logging, it is reported and the wait continues.
FATAL_RE = re.compile(
    r"EngineDeadError|EngineCore encountered an issue|Worker failed with error"
    r"|Engine (?:core )?initialization failed|Engine process failed to start"
    r"|CUDA error|cudaError|CUDA out of memory|OutOfMemoryError"
    r"|Address already in use|No CUDA GPUs are available"
    r"|NCCL error|ncclUnhandledCudaError|ncclSystemError|ncclInternalError"
    r"|ncclInvalidUsage|IBV_WC_RETRY_EXC_ERR|ss1\.ss_family == ss2\.ss_family"
    r"|serve: error:|unrecognized arguments|cannot be converted"
    r"|Free memory on device .* is less than desired"
    r"|ModuleNotFoundError|ImportError: |FileNotFoundError"
    r"|Traceback \(most recent call last\)",
    re.I)
_CONFIRM_SEC = 60.0        # after a fatal match: exit or silence => dead

_POLL = 5.0


def _clean(text: str) -> list[str]:
    out = []
    for line in re.split(r"[\r\n]+", _ANSI.sub("", text)):
        line = line.rstrip()
        if line.strip():
            out.append(line[:400])
    return out


async def _docker(*args: str, timeout: float = 10.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return -1, ""
    return proc.returncode or 0, out.decode(errors="replace")


async def head_container(recipe_id: str) -> str | None:
    """The head (serve) container of a job on this node, if it exists."""
    rc, out = await _docker("ps", "--format", "{{.Names}}",
                            "--filter", f"name=sparkrun_{recipe_id}")
    names = out.split()
    if rc != 0 or not names:
        return None
    for suffix in ("_head", "_node_0", "_solo"):
        for n in names:
            if n.endswith(suffix):
                return n
    return names[0]


async def serve_alive(container: str) -> bool | None:
    """Is the serve process inside `container` alive? None when unknowable
    (older sparkrun without the pid file, or the container is gone)."""
    rc, out = await _docker("exec", container, "cat", SERVE_PID)
    pid = out.strip()
    if rc != 0 or not pid.isdigit():
        return None
    rc, _ = await _docker("exec", container, "kill", "-0", pid)
    return rc == 0


async def log_size(container: str) -> int | None:
    rc, out = await _docker("exec", container, "stat", "-c", "%s", SERVE_LOG)
    return int(out) if rc == 0 and out.strip().isdigit() else None


async def read_from(container: str, offset: int, limit: int = 65536) -> str:
    """Bytes of the serve log after `offset` (capped)."""
    rc, out = await _docker(
        "exec", container, "sh", "-c",
        f"tail -c +{offset + 1} {SERVE_LOG} | head -c {limit}")
    return out if rc == 0 else ""


async def tail_log(container: str, lines: int = 40) -> list[str]:
    rc, out = await _docker("exec", container, "tail", "-n", str(lines), SERVE_LOG)
    return _clean(out) if rc == 0 else []


async def save_logs(recipe_id: str, label: str | None = None) -> str | None:
    """Capture every rank's log (`sparkrun logs -a`) to a file under
    DATA_DIR/launch-logs. Returns the path, or None when nothing was read."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", label or recipe_id).strip("-")
        path = f"{LOG_DIR}/{stamp}-{base}-{recipe_id[:12]}.log"
        proc = await asyncio.create_subprocess_exec(
            "sparkrun", "logs", recipe_id, "--cluster", config.SPARKRUN_CLUSTER,
            "-a", "-n", "500",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=90.0)
        except asyncio.TimeoutError:
            proc.kill()
            return None
        text = _ANSI.sub("", out.decode(errors="replace"))
        if not text.strip():
            return None
        with open(path, "w") as fh:
            fh.write(text)
        return path
    except Exception:
        return None


async def _health(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(f"{config.VLLM_BASE}/health")
        return r.status_code == 200
    except Exception:
        return False


async def watch(recipe_id: str, emit, touch, is_running) -> tuple[bool, str | None, list[str]]:
    """Follow one launch until vLLM is healthy or demonstrably dead.

    emit(line) streams serve-log lines and verdicts into the operation
    output; touch(**kw) updates its phase/message; is_running() is an async
    callable answering whether sparkrun still lists the job.

    Returns (ok, error, excerpt): error is a one-line verdict and excerpt the
    last serve-log lines, for the operation record and the alert.
    """
    stall = config.LAUNCH_STALL_MIN * 60
    hard = config.LAUNCH_MAX_MIN * 60
    t0 = time.time()
    container: str | None = None
    offset = 0
    last_growth = t0
    fatal_at: float | None = None
    fatal_line: str | None = None
    container_seen = False
    missing_polls = 0

    async def fail(error: str) -> tuple[bool, str, list[str]]:
        excerpt = await tail_log(container, 40) if container else []
        emit(f"[sparkdash] LAUNCH FAILED: {error}")
        return False, error, excerpt

    touch(phase="loading", message="Waiting for vLLM to come up (watching the serve log)…")
    async with httpx.AsyncClient(timeout=4.0) as client:
        while True:
            if await _health(client):
                return True, None, []
            now = time.time()
            elapsed = (now - t0) / 60

            if container is None:
                container = await head_container(recipe_id)
                if container:
                    container_seen = True
                    missing_polls = 0
                    emit(f"[sparkdash] following {SERVE_LOG} in {container}")
                else:
                    missing_polls += 1
                    # The head container vanished (sparkrun's in-container
                    # watchdog kills PID 1 when the serve process dies).
                    if container_seen or missing_polls >= 6:
                        if not await is_running():
                            return await fail(
                                "the job is no longer running — its containers "
                                "exited" if container_seen else
                                "sparkrun run returned but no job is running")
            else:
                size = await log_size(container)
                if size is None:
                    # exec failed: container gone? re-resolve next poll
                    if not await head_container(recipe_id):
                        container = None
                        continue
                else:
                    if size > offset:
                        chunk = await read_from(container, offset)
                        offset += len(chunk.encode(errors="replace"))
                        last_growth = now
                        lines = _clean(chunk)
                        for line in lines[-25:]:
                            emit(line)
                        for line in lines:
                            if fatal_at is None and FATAL_RE.search(line):
                                fatal_at, fatal_line = now, line
                                emit(f"[sparkdash] error in serve log — confirming "
                                     f"for {int(_CONFIRM_SEC)}s: {line[:160]}")
                    elif size < offset:
                        offset = 0   # log was truncated/recreated
                alive = await serve_alive(container)
                if alive is False:
                    return await fail(
                        "the vLLM process exited inside the head container "
                        "(the container itself is still up)"
                        + (f" — last error: {fatal_line[:200]}" if fatal_line else ""))
                if fatal_at is not None and now - fatal_at >= _CONFIRM_SEC:
                    if last_growth <= fatal_at + 1.0:
                        return await fail(
                            f"vLLM logged a fatal error and went silent: {fatal_line[:200]}")
                    emit("[sparkdash] vLLM is still logging after the error above; "
                         "continuing to wait")
                    fatal_at = None

            if now - last_growth > stall:
                return await fail(
                    f"no serve-log output for {int(stall // 60)} min and /health "
                    f"still failing ({elapsed:.0f} min since launch) — hung"
                    + (f"; last error: {fatal_line[:200]}" if fatal_line else ""))
            if now - t0 > hard:
                return await fail(
                    f"vLLM still not healthy after {int(hard // 60)} min")

            touch(message=f"Loading… {elapsed:.0f} min, log "
                          f"{'growing' if now - last_growth < 30 else 'quiet ' + str(int((now - last_growth) // 60)) + ' min'}")
            await asyncio.sleep(_POLL)
