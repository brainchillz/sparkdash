"""Recipe lifecycle control: start / stop / restart (Phase 3).

Wraps sparkrun's own tooling:
  * start   : `sparkrun run <recipe> (--solo | --cluster <name>) --no-follow`
  * stop    : `sparkrun stop <id> --cluster <name>`
  * restart : `export running-recipe` (captures the exact effective config) then
              stop, then re-run that file with the same topology
  * preflight: `sparkrun run … --dry-run` → VRAM fit estimate, no side effects

Start is long-running (image start, then vLLM loads weights), so operations run
as a tracked job with streamed output; readiness is confirmed by polling vLLM's
/health. One operation at a time.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time

import httpx

from . import config
from .collectors import _parse_status

_HEALTH_TIMEOUT = 20 * 60   # allow big models time to load weights
_MAX_OUTPUT = 500

# sparkrun labels a workload with the recipe argument, so restart re-runs from a
# file named after the model to keep the dashboard name clean and stable.
_RECIPE_TMP = "/tmp/sparkdash-recipes"

# Durable copies of effective recipes captured on stop/restart, so a recipe that
# isn't in any registry (e.g. a custom/local one) can always be re-run.
_SAVED_DIR = str(config.DATA_DIR / "recipes")
_LAST_FILE = str(config.DATA_DIR / "last_recipe.json")


def _save_last(name: str, path: str, mode: str, model: str) -> None:
    try:
        with open(_LAST_FILE, "w") as fh:
            json.dump({"name": name, "path": path, "mode": mode,
                       "model": model, "saved_at": time.time()}, fh)
    except OSError:
        pass


def last_recipe() -> dict:
    """The most recently stopped recipe, if its saved file still exists."""
    try:
        with open(_LAST_FILE) as fh:
            info = json.load(fh)
        if info.get("path") and os.path.isfile(info["path"]):
            return info
    except (OSError, ValueError):
        pass
    return {}


def saved_recipes() -> list[dict]:
    """Durably-saved effective recipes (custom recipes not in any registry)."""
    out = []
    try:
        for fn in sorted(os.listdir(_SAVED_DIR)):
            if fn.startswith(".export_") or not fn.endswith((".yaml", ".yml")):
                continue
            path = f"{_SAVED_DIR}/{fn}"
            out.append({"name": re.sub(r"\.ya?ml$", "", fn), "path": path,
                        "model": _read_model(path)})
    except OSError:
        pass
    return out


def _read_model(path: str) -> str:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("model:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _clean_base(model: str, fallback: str | None) -> str:
    src = (model.split("/")[-1] if model else "") or (fallback or "") or "recipe"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", src).strip("-") or "recipe"


def _mode_flags(mode: str) -> list[str]:
    if mode == "solo":
        return ["--solo"]
    return ["--cluster", config.SPARKRUN_CLUSTER]


async def current_recipe() -> dict:
    """Parse `sparkrun status` into the running recipe (id, name, mode)."""
    proc = await asyncio.create_subprocess_exec(
        "sparkrun", "status", "--cluster", config.SPARKRUN_CLUSTER,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=25.0)
    rec = _parse_status(out.decode(errors="replace"))
    if rec.get("running"):
        rec["mode"] = "cluster" if len(rec.get("containers", [])) > 1 else "solo"
    return rec


# -- dry-run pre-flight ------------------------------------------------------

async def preflight(recipe: str, mode: str) -> dict:
    argv = ["sparkrun", "run", recipe, *_mode_flags(mode), "--dry-run"]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
    text = out.decode(errors="replace")

    def grab(pattern: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else None

    err = None
    if proc.returncode != 0 or "Error" in text:
        err = grab(r"Error:\s*(.+)") or "recipe check failed"
    return {
        "ok": proc.returncode == 0 and err is None,
        "error": err,
        "fit": grab(r"DGX Spark fit:\s*(YES|NO)"),
        "mode": grab(r"Mode:\s*(.+)"),
        "weights": grab(r"Model weights:\s*([\d.]+ [A-Z]+)"),
        "per_gpu": grab(r"Per-GPU total:\s*([\d.]+ [A-Z]+)"),
        "context_mult": grab(r"Context multiplier:\s*([\d.]+x)"),
    }


class RecipeController:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._op: dict = {"active": False, "action": None, "phase": "idle",
                          "output": []}

    def status(self) -> dict:
        s = dict(self._op)
        s["output"] = s.get("output", [])[-250:]  # cap payload
        return s

    def _busy(self) -> bool:
        return self._task is not None and not self._task.done()

    # -- public entry points --------------------------------------------------

    def start(self, recipe: str, mode: str) -> None:
        if self._busy():
            raise RuntimeError("An operation is already running")
        self._begin("start", recipe, mode)
        self._task = asyncio.create_task(self._do_start(recipe, mode))

    def stop(self) -> None:
        if self._busy():
            raise RuntimeError("An operation is already running")
        self._begin("stop", None, None)
        self._task = asyncio.create_task(self._do_stop())

    def restart(self, mode: str | None = None) -> None:
        if self._busy():
            raise RuntimeError("An operation is already running")
        self._begin("restart", None, mode)
        self._task = asyncio.create_task(self._do_restart(mode))

    # -- state helpers --------------------------------------------------------

    def _begin(self, action, recipe, mode) -> None:
        self._op = {
            "active": True, "action": action, "phase": "starting",
            "recipe": recipe, "mode": mode, "output": [],
            "message": "", "error": None,
            "started": time.time(), "updated": time.time(),
        }

    def _touch(self, **kw) -> None:
        self._op.update(kw)
        self._op["updated"] = time.time()

    def _emit(self, line: str) -> None:
        out = self._op.setdefault("output", [])
        out.append(line)
        del out[:-_MAX_OUTPUT]
        self._op["updated"] = time.time()

    def _finish(self, ok: bool, message: str, error: str | None = None) -> None:
        self._touch(active=False, phase=("done" if ok else "error"),
                    message=message, error=error)

    async def _stream(self, argv: list[str]) -> int:
        """Run a command, streaming its output into the op; return exit code."""
        self._emit(f"$ {' '.join(argv)}")
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        assert proc.stdout is not None
        async for raw in proc.stdout:
            self._emit(raw.decode(errors="replace").rstrip("\n"))
        return await proc.wait()

    async def _wait_healthy(self) -> bool:
        self._touch(phase="loading", message="Waiting for vLLM to become healthy…")
        deadline = time.time() + _HEALTH_TIMEOUT
        async with httpx.AsyncClient(timeout=4.0) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"{config.VLLM_BASE}/health")
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(5.0)
        return False

    # -- operations -----------------------------------------------------------

    async def _do_start(self, recipe: str, mode: str) -> None:
        try:
            self._touch(phase="launching", message=f"Starting {recipe} ({mode})")
            rc = await self._stream(
                ["sparkrun", "run", recipe, *_mode_flags(mode), "--no-follow"])
            if rc != 0:
                self._finish(False, "Start failed", f"sparkrun run exited {rc}")
                return
            if await self._wait_healthy():
                self._finish(True, "Recipe running — vLLM healthy")
            else:
                self._finish(False, "Started, but vLLM did not become healthy in time",
                             "health timeout")
        except Exception as exc:
            self._finish(False, "Start failed", f"{type(exc).__name__}: {exc}")

    async def _do_stop(self) -> None:
        try:
            rec = await current_recipe()
            if not rec.get("running"):
                self._finish(True, "Nothing was running")
                return
            # Capture the effective recipe first so it can be re-run later even
            # if it isn't in any registry (best-effort; never blocks the stop).
            await self._preserve(rec)
            self._touch(phase="stopping", recipe=rec.get("name"),
                        message=f"Stopping {rec.get('name')}")
            rc = await self._stream(
                ["sparkrun", "stop", rec["id"], "--cluster", config.SPARKRUN_CLUSTER])
            if rc != 0:
                self._finish(False, "Stop failed", f"sparkrun stop exited {rc}")
                return
            self._finish(True, "Recipe stopped")
        except Exception as exc:
            self._finish(False, "Stop failed", f"{type(exc).__name__}: {exc}")

    async def _preserve(self, rec: dict) -> None:
        """Export the running recipe to a durable file and record it as 'last'."""
        try:
            os.makedirs(_SAVED_DIR, exist_ok=True)
            raw = f"{_SAVED_DIR}/.export_{secrets.token_hex(4)}.yaml"
            self._touch(message="Saving effective recipe…")
            rc = await self._stream(
                ["sparkrun", "export", "running-recipe", rec["id"],
                 "--cluster", config.SPARKRUN_CLUSTER, "--save", raw])
            if rc != 0 or not os.path.isfile(raw):
                return
            model = _read_model(raw)
            base = _clean_base(model, rec.get("name"))
            path = f"{_SAVED_DIR}/{base}.yaml"
            os.replace(raw, path)
            _save_last(base, path, rec.get("mode") or "cluster", model)
            self._emit(f"[sparkdash] saved recipe -> {path}")
        except Exception as exc:  # never let saving block a stop
            self._emit(f"[sparkdash] could not save recipe: {exc}")

    async def _do_restart(self, mode: str | None) -> None:
        try:
            rec = await current_recipe()
            if not rec.get("running"):
                self._finish(False, "Nothing is running to restart", "not running")
                return
            use_mode = mode or rec.get("mode") or "cluster"
            os.makedirs(_RECIPE_TMP, exist_ok=True)
            raw = f"{_RECIPE_TMP}/.export_{secrets.token_hex(4)}.yaml"
            self._touch(phase="exporting", recipe=rec.get("name"), mode=use_mode,
                        message="Capturing effective recipe")
            rc = await self._stream(
                ["sparkrun", "export", "running-recipe", rec["id"],
                 "--cluster", config.SPARKRUN_CLUSTER, "--save", raw])
            if rc != 0:
                self._finish(False, "Restart failed (export)", f"export exited {rc}")
                return
            # Re-run from a file named after the model so the job name stays clean.
            base = _clean_base(_read_model(raw), rec.get("name"))
            tmp = f"{_RECIPE_TMP}/{base}.yaml"
            os.replace(raw, tmp)

            self._touch(phase="stopping", message=f"Stopping {rec.get('name')}")
            rc = await self._stream(
                ["sparkrun", "stop", rec["id"], "--cluster", config.SPARKRUN_CLUSTER])
            if rc != 0:
                self._finish(False, "Restart failed (stop)", f"stop exited {rc}")
                return

            self._touch(phase="launching", message="Re-launching recipe")
            rc = await self._stream(
                ["sparkrun", "run", tmp, *_mode_flags(use_mode), "--no-follow"])
            if rc != 0:
                self._finish(False, "Restart failed (run)", f"run exited {rc}")
                return
            if await self._wait_healthy():
                self._finish(True, "Recipe restarted — vLLM healthy")
            else:
                self._finish(False, "Restarted, but vLLM did not become healthy",
                             "health timeout")
        except Exception as exc:
            self._finish(False, "Restart failed", f"{type(exc).__name__}: {exc}")


controller = RecipeController()
