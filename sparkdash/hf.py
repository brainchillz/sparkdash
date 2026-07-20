"""HuggingFace model preloading (Phase 2, first write feature).

Rather than reimplement downloading + cross-node copying, this drives
sparkrun's own model functions with sparkrun's interpreter:

  * download_model(model_id)          - single-node preload onto the head
  * distribute_model_from_head(...)    - download on head, then rsync to the
                                         worker(s) over the 200GbE RDMA link
                                         (passed as worker_transfer_hosts)

so a preload behaves exactly like the model staging a cluster recipe run does.
On top of that we track progress by disk usage vs. the repo's known size (from
the HF API) — identical for the head (local `du`) and workers (remote `du`).
One job runs at a time.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import time
from pathlib import Path

from . import config

HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

_HEAD = next(n for n in config.NODES if n["local"])
_WORKERS = [n for n in config.NODES if not n["local"]]


def repo_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def dirname_to_repo(dirname: str) -> str:
    return dirname.removeprefix("models--").replace("--", "/")


def _size_patterns(model_ref: str) -> tuple[str, list[str] | None]:
    """(repo_id, patterns) for sizing. Handles the GGUF `repo:QUANT` form."""
    if ":" in model_ref:
        repo, tag = model_ref.rsplit(":", 1)
        return repo, [f"*{tag.lower()}*"]
    return model_ref, None


# -- size resolution (progress denominator) ---------------------------------

def resolve_size(model_ref: str) -> tuple[int, int]:
    from huggingface_hub import HfApi
    repo_id, patterns = _size_patterns(model_ref)
    info = HfApi().model_info(repo_id, files_metadata=True)
    files = info.siblings or []
    if patterns:
        files = [s for s in files
                 if any(fnmatch.fnmatch(s.rfilename.lower(), p) for p in patterns)]
    return sum(s.size or 0 for s in files), len(files)


# -- cache inventory ---------------------------------------------------------

_LIST_CMD = (
    'for d in ~/.cache/huggingface/hub/models--*; do [ -d "$d" ] && '
    'echo "$(basename "$d") $(du -sb "$d" 2>/dev/null | cut -f1)"; done'
)


async def _run(argv: list[str], timeout: float) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return out.decode(errors="replace")


def _ssh(ip: str, cmd: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
            f"{config.SSH_USER}@{ip}", cmd]


async def _du_list(node: dict) -> dict[str, int]:
    argv = ["bash", "-c", _LIST_CMD] if node["local"] else _ssh(node["ip"], _LIST_CMD)
    result: dict[str, int] = {}
    for line in (await _run(argv, 30.0)).splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


async def list_cache() -> dict:
    """Cached models across all nodes: repo -> per-node sizes."""
    per_node = await asyncio.gather(*(_du_list(n) for n in config.NODES))
    models: dict[str, dict] = {}
    for node, sizes in zip(config.NODES, per_node):
        for dirname, size in sizes.items():
            m = models.setdefault(dirname, {
                "repo": dirname_to_repo(dirname), "sizes": {}})
            m["sizes"][node["name"]] = size
    return {"nodes": [n["name"] for n in config.NODES],
            "models": sorted(models.values(), key=lambda m: m["repo"].lower())}


_DIRNAME_RE = re.compile(r"^models--[A-Za-z0-9._-]+$")


async def delete_cached(repo_id: str) -> dict:
    """Delete a cached model dir on every node. Refuses the loaded model."""
    from .chat import current_model_id
    if (await current_model_id()) == repo_id:
        raise ValueError("that model is currently loaded — stop the recipe first")
    dirname = repo_dirname(repo_id)
    if not _DIRNAME_RE.match(dirname):
        raise ValueError("invalid model id")
    # $HOME expands inside double quotes; dirname is validated to be safe.
    cmd = f'rm -rf "$HOME/.cache/huggingface/hub/{dirname}"'
    results = {}
    for node in config.NODES:
        argv = ["bash", "-c", cmd] if node["local"] else _ssh(node["ip"], cmd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            rc = await asyncio.wait_for(proc.wait(), timeout=60.0)
            results[node["name"]] = rc == 0
        except Exception:
            results[node["name"]] = False
    return results


async def node_free_bytes(node: dict, path: str = "~/.cache/huggingface") -> int:
    """Free bytes on the filesystem holding `path` (local or over SSH)."""
    cmd = f"df -B1 --output=avail {path} 2>/dev/null | tail -1"
    argv = ["bash", "-c", cmd] if node["local"] else _ssh(node["ip"], cmd)
    try:
        s = (await _run(argv, 20.0)).strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return 0


# Copies refuse to start unless the target has the model's size plus this
# safety margin free — better a clear refusal than a partial 100 GB copy.
SPACE_MARGIN = 0.02


async def space_error(dirname: str, total: int, nodes: list[dict]) -> str | None:
    """Human-readable error if any node lacks room for the rest of the model."""
    for node in nodes:
        have = await _du_bytes(node, dirname)
        need = int((total - have) * (1 + SPACE_MARGIN))
        if need <= 0:
            continue
        free = await node_free_bytes(node)
        if free < need:
            return (f"not enough space on {node['name']}: needs "
                    f"~{need / 1e9:.1f} GB free, has {free / 1e9:.1f} GB")
    return None


async def _du_bytes(node: dict, dirname: str) -> int:
    path = f"~/.cache/huggingface/hub/{dirname}"
    cmd = f'du -sb {path} 2>/dev/null | cut -f1'
    argv = ["bash", "-c", cmd] if node["local"] else _ssh(node["ip"], cmd)
    try:
        s = (await _run(argv, 20.0)).strip()
        return int(s) if s.isdigit() else 0
    except Exception:
        return 0


# -- preload job -------------------------------------------------------------

class Preloader:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel = False
        self._state: dict = {"active": False, "phase": "idle"}

    def status(self) -> dict:
        return dict(self._state)

    def start(self, model_ref: str, mirror: bool, existing: bool = False) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("A preload is already running")
        self._cancel = False
        self._state = {
            "active": True, "phase": "resolving", "model": model_ref,
            "mirror": mirror and bool(_WORKERS),
            "total_bytes": 0, "head_bytes": 0, "workers": {},
            "message": "Resolving size…", "error": None,
            "started": time.time(), "updated": time.time(),
        }
        self._task = asyncio.create_task(self._drive(model_ref, mirror, existing))

    async def cancel(self) -> None:
        self._cancel = True
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()

    def _touch(self, **kw) -> None:
        self._state.update(kw)
        self._state["updated"] = time.time()

    async def _drive(self, model_ref: str, mirror: bool, existing: bool = False) -> None:
        try:
            do_mirror = mirror and bool(_WORKERS)
            if existing:
                # Already downloaded: size from local cache, no HF API, no download.
                total = await _du_bytes(_HEAD, repo_dirname(_size_patterns(model_ref)[0]))
                if total <= 0:
                    self._touch(active=False, phase="error", message="Failed",
                                error=f"{model_ref} is not cached on {_HEAD['name']}")
                    return
                if not do_mirror:
                    self._touch(active=False, phase="done", total_bytes=total,
                                head_bytes=total, message="Nothing to mirror")
                    return
                dirname = repo_dirname(_size_patterns(model_ref)[0])
                space = await space_error(dirname, total, _WORKERS)
                if space:
                    self._touch(active=False, phase="error", message="Failed",
                                error=space)
                    return
                self._touch(total_bytes=total, head_bytes=total, phase="mirroring",
                            message="Mirroring to cluster over RDMA")
            else:
                total, nfiles = await asyncio.to_thread(resolve_size, model_ref)
                dirname = repo_dirname(_size_patterns(model_ref)[0])
                targets = [_HEAD] + (_WORKERS if do_mirror else [])
                space = await space_error(dirname, total, targets)
                if space:
                    self._touch(active=False, phase="error", message="Failed",
                                error=space)
                    return
                self._touch(total_bytes=total, file_count=nfiles,
                            phase="downloading",
                            message=f"Downloading {nfiles} file(s) to {_HEAD['name']}")

            argv = self._build_cmd(model_ref, do_mirror)
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            await self._monitor(model_ref, total, do_mirror)
            rc = await self._proc.wait()

            if self._cancel:
                self._touch(active=False, phase="cancelled", message="Cancelled")
                return
            if rc != 0:
                err = (await self._proc.stderr.read()).decode(errors="replace")
                raise RuntimeError(err.strip()[-500:] or f"exit {rc}")

            workers = {w["name"]: total for w in _WORKERS} if do_mirror else {}
            self._touch(active=False, phase="done", head_bytes=total,
                        workers=workers, message="Model ready")
        except Exception as exc:
            self._touch(active=False, phase="error",
                        error=f"{type(exc).__name__}: {exc}", message="Failed")

    def _build_cmd(self, model_ref: str, do_mirror: bool) -> list[str]:
        """A sparkrun-interpreter subprocess that reuses sparkrun's own code."""
        if do_mirror:
            hosts = [_HEAD["ip"]] + [w["ip"] for w in _WORKERS]
            transfer = [config.RDMA_IP.get(w["name"], w["ip"]) for w in _WORKERS]
            script = (
                "import sys, json\n"
                "from sparkrun.models.distribute import distribute_model_from_head\n"
                "fails = distribute_model_from_head(\n"
                "    model_id=sys.argv[1], hosts=json.loads(sys.argv[2]),\n"
                "    worker_transfer_hosts=json.loads(sys.argv[3]),\n"
                "    ssh_user=sys.argv[4])\n"
                "sys.exit(3 if fails else 0)\n"
            )
            return [config.SPARKRUN_PYTHON, "-c", script, model_ref,
                    json.dumps(hosts), json.dumps(transfer), config.SSH_USER]
        script = (
            "import sys\n"
            "from sparkrun.models.download import download_model\n"
            "sys.exit(download_model(sys.argv[1]))\n"
        )
        return [config.SPARKRUN_PYTHON, "-c", script, model_ref]

    async def _monitor(self, model_ref: str, total: int, do_mirror: bool) -> None:
        dirname = repo_dirname(_size_patterns(model_ref)[0])
        while self._proc.returncode is None:
            if self._cancel:
                return
            head = await _du_bytes(_HEAD, dirname)
            self._state["head_bytes"] = head
            head_done = total and head >= total * 0.999
            if do_mirror:
                for w in _WORKERS:
                    wb = await _du_bytes(w, dirname)
                    self._state["workers"][w["name"]] = wb
                if head_done:
                    self._touch(phase="mirroring",
                                message="Mirroring to cluster over RDMA")
                else:
                    self._touch()
            else:
                self._touch()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass


preloader = Preloader()
