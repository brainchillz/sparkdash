"""Live container-log streaming for the running recipe.

The **vLLM** server output goes to a file inside the recipe's head container
(`sparkrun` launches `vllm serve … > /tmp/sparkrun_serve.log`), not the
container's stdout — so `docker logs` only shows Ray. We therefore
`docker exec … tail -f` that file. If it's absent (non-vLLM runtime), we fall
back to `docker logs` (Ray/container stdout).

`docker exec tail -f` leaves the in-container `tail` running if only the local
client is killed, so each stream records its tail PID to a per-connection
marker file and kills exactly that process in `stop_stream()`. Cleanup is a
plain coroutine (not an async-generator finally) so the caller can `shield` it
and guarantee it completes even when the connection is being torn down.
"""

from __future__ import annotations

import asyncio
import re
import secrets

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
SERVE_LOG = "/tmp/sparkrun_serve.log"


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def decode_line(raw: bytes) -> str:
    return strip_ansi(raw.decode(errors="replace").rstrip("\n"))


async def find_container(recipe_id: str) -> str | None:
    """Return the head container name for a recipe id (falls back to any match)."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "--format", "{{.Names}}",
        "--filter", f"name=sparkrun_{recipe_id}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    names = out.decode(errors="replace").split()
    if not names:
        return None
    for n in names:
        if n.endswith("_head"):
            return n
    return names[0]


async def _has_serve_log(container: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", container, "test", "-f", SERVE_LOG,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    try:
        return await asyncio.wait_for(proc.wait(), timeout=8.0) == 0
    except asyncio.TimeoutError:
        return False


class Stream:
    """A running log tail: the subprocess plus what's needed to clean it up."""

    def __init__(self, proc, marker: str | None, container: str) -> None:
        self.proc = proc
        self.marker = marker          # set when tailing the in-container serve log
        self.container = container


async def start_stream(container: str, tail: int = 400) -> Stream:
    """Start following the vLLM serve log (preferred) or the container stdout."""
    if await _has_serve_log(container):
        marker = f"/tmp/.sparkdash_tail.{secrets.token_hex(6)}"
        # `exec tail` inherits the shell's PID ($$), so the marker holds tail's PID.
        argv = ["docker", "exec", container, "sh", "-c",
                f"echo $$ > {marker}; exec tail -n {tail} -f {SERVE_LOG}"]
    else:
        marker = None
        argv = ["docker", "logs", "--tail", str(tail), "-f", container]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    return Stream(proc, marker, container)


async def stop_stream(s: Stream) -> None:
    """Kill exactly this stream's in-container tail (if any) and the subprocess.

    Safe to `asyncio.shield()` — it does not depend on the caller staying alive.
    """
    if s.marker:
        kill = await asyncio.create_subprocess_exec(
            "docker", "exec", s.container, "sh", "-c",
            f"kill $(cat {s.marker} 2>/dev/null) 2>/dev/null; rm -f {s.marker}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            await asyncio.wait_for(kill.wait(), timeout=6.0)
        except asyncio.TimeoutError:
            pass
    if s.proc.returncode is None:
        s.proc.terminate()
        try:
            await asyncio.wait_for(s.proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            s.proc.kill()
