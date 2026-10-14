"""One operation at a time, whichever engine it drives.

Recipe (sparkrun) operations and stack operations share this single state:
the admin UI has one progress panel that shows whichever is running, the
watchdog can tell "someone is already restarting it" from "it is dead", and
two engines can never be started on top of each other by racing requests.
"""

from __future__ import annotations

import asyncio
import time

_MAX_OUTPUT = 500


class OpState:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._op: dict = {"active": False, "action": None, "phase": "idle",
                          "output": []}

    # -- queries --------------------------------------------------------------

    def status(self) -> dict:
        s = dict(self._op)
        s["output"] = s.get("output", [])[-250:]  # cap payload
        return s

    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def wait(self) -> None:
        """Wait for the current operation (if any) to finish."""
        if self._task is not None and not self._task.done():
            try:
                await asyncio.shield(self._task)
            except Exception:
                pass

    # -- lifecycle ------------------------------------------------------------

    def launch(self, action: str, coro, **fields) -> None:
        """Begin `action` and run `coro` (a coroutine object) as its task.
        Raises RuntimeError when another operation is still running."""
        if self.busy():
            coro.close()
            raise RuntimeError("An operation is already running")
        self.begin(action, **fields)
        self._task = asyncio.create_task(coro)

    def begin(self, action: str, **fields) -> None:
        self._op = {
            "active": True, "action": action, "phase": "starting",
            "engine": None, "recipe": None, "mode": None, "output": [],
            "message": "", "error": None,
            "started": time.time(), "updated": time.time(),
        }
        self._op.update(fields)

    def touch(self, **kw) -> None:
        self._op.update(kw)
        self._op["updated"] = time.time()

    def emit(self, line: str) -> None:
        out = self._op.setdefault("output", [])
        out.append(line)
        del out[:-_MAX_OUTPUT]
        self._op["updated"] = time.time()

    def finish(self, ok: bool, message: str, error: str | None = None) -> None:
        self.touch(active=False, phase=("done" if ok else "error"),
                   message=message, error=error)

    async def stream(self, argv: list[str], env: dict | None = None) -> int:
        """Run a command, streaming its output into the op; return exit code."""
        self.emit(f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env)
        assert proc.stdout is not None
        async for raw in proc.stdout:
            self.emit(raw.decode(errors="replace").rstrip("\n"))
        return await proc.wait()


state = OpState()
