"""Model backup / restore to a shared (typically NFS) location.

A cached HuggingFace model on this cluster is the same bytes on every node —
the Mirror feature keeps the per-node caches identical — so a backup stores a
single canonical copy rather than one per host. Layout:

    <base>/Sparkdash/Models/models--<org>--<repo>/
        <blobs, snapshots, refs …>          (the HF hub dir, verbatim)
        sparkdash-backup.json               (manifest: repo, size, recipes, …)

Keeping the HuggingFace `models--…` directory name means a restore is just a
copy back into each node's `~/.cache/huggingface/hub/`. The manifest records
which recipe(s) use the model so a backup is self-describing.

Backup copies from whichever node has the model (preferring the head) to the
share. Restore copies the backup onto the head, then re-mirrors to the
worker(s) over RDMA by reusing the existing preload/mirror path. One backup or
restore runs at a time.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from pathlib import Path

from . import config, hf, recipe_ops
from .chat import current_model_id

_HEAD = hf._HEAD
_WORKERS = hf._WORKERS


def _root(base: str) -> Path:
    return Path(base) / config.BACKUP_SUBDIR


# -- helpers -----------------------------------------------------------------

async def _du_local(path: Path) -> int:
    cmd = f"du -sb {shlex.quote(str(path))} 2>/dev/null | cut -f1"
    out = (await hf._run(["bash", "-c", cmd], 120.0)).strip()
    return int(out) if out.isdigit() else 0


async def _mkdirp(path: Path) -> None:
    await asyncio.to_thread(os.makedirs, path, exist_ok=True)


async def _source_node(dirname: str) -> dict | None:
    """First node that actually has the model cached (head preferred)."""
    for node in [_HEAD, *_WORKERS]:
        if await hf._du_bytes(node, dirname) > 0:
            return node
    return None


async def recipes_for_model(repo: str) -> list[str]:
    """Recipe name(s) whose model is this repo — registry + saved custom."""
    names: list[str] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            "sparkrun", "list", "--json",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        for r in json.loads(out.decode() or "[]"):
            if (r.get("model") or "").strip() == repo and r.get("name"):
                names.append(r["name"])
    except Exception:
        pass
    for r in recipe_ops.saved_recipes():
        if (r.get("model") or "").strip() == repo and r.get("name"):
            names.append(r["name"])
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def _read_manifest(d: Path) -> dict:
    try:
        return json.loads((d / config.BACKUP_MANIFEST).read_text())
    except Exception:
        return {}


def _writable(base: str) -> bool:
    try:
        p = Path(base)
        return p.is_dir() and os.access(p, os.W_OK)
    except Exception:
        return False


def _disk_stats(path: str) -> dict | None:
    try:
        u = shutil.disk_usage(path)
        return {"total": u.total, "free": u.free}
    except Exception:
        return None


def _free_bytes(path: str) -> int:
    stats = _disk_stats(path)
    return stats["free"] if stats else 0


async def list_backups(base: str) -> dict:
    """Backups present at <base>/Sparkdash/Models, annotated with cache state."""
    root = _root(base)
    exists = await asyncio.to_thread(root.is_dir)
    try:
        cache = await hf.list_cache()
        cached = {m["repo"] for m in cache["models"]}
    except Exception:
        cached = set()
    items: list[dict] = []
    if exists:
        for d in sorted(root.glob("models--*")):
            if not d.is_dir() or not hf._DIRNAME_RE.match(d.name):
                continue
            repo = hf.dirname_to_repo(d.name)
            man = _read_manifest(d)
            size = man.get("size_bytes")
            if not size:
                size = await _du_local(d)
            items.append({
                "repo": repo, "dirname": d.name, "size": size,
                "recipes": man.get("recipes", []),
                "source_node": man.get("source_node"),
                "saved_at": man.get("saved_at"),
                "cached": repo in cached,
            })
    return {"base": base, "root": str(root), "exists": exists,
            "writable": _writable(base), "disk": _disk_stats(base),
            "backups": items}


async def delete_backup(repo: str, base: str) -> None:
    """Remove one model's backup dir from the share."""
    dirname = hf.repo_dirname(repo)
    if not hf._DIRNAME_RE.match(dirname):
        raise ValueError("invalid model id")
    d = _root(base) / dirname
    if not await asyncio.to_thread(d.is_dir):
        raise ValueError("no backup found at that location")
    await asyncio.to_thread(shutil.rmtree, d)


# -- backup / restore job ----------------------------------------------------

class BackupManager:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel = False
        self._state: dict = {"active": False, "phase": "idle"}

    def status(self) -> dict:
        return dict(self._state)

    def _busy(self) -> bool:
        return bool(self._task and not self._task.done())

    def _init(self, op: str, repo: str, message: str) -> None:
        self._cancel = False
        self._state = {
            "active": True, "op": op, "phase": "preparing", "model": repo,
            "total_bytes": 0, "done_bytes": 0, "message": message,
            "error": None, "mirroring": False,
            "started": time.time(), "updated": time.time(),
        }

    def _touch(self, **kw) -> None:
        self._state.update(kw)
        self._state["updated"] = time.time()

    async def cancel(self) -> None:
        self._cancel = True
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()

    def start_backup(self, repo: str, base: str) -> None:
        if self._busy():
            raise RuntimeError("A backup or restore is already running")
        self._init("backup", repo, "Preparing backup…")
        self._task = asyncio.create_task(self._backup(repo, base))

    def start_restore(self, repo: str, base: str, mirror: bool) -> None:
        if self._busy():
            raise RuntimeError("A backup or restore is already running")
        self._init("restore", repo, "Preparing restore…")
        self._task = asyncio.create_task(self._restore(repo, base, mirror))

    def start_verify(self, repo: str, base: str) -> None:
        if self._busy():
            raise RuntimeError("A backup or restore is already running")
        self._init("verify", repo, "Preparing verification…")
        self._task = asyncio.create_task(self._verify(repo, base))

    # rsync flags: recurse, copy symlinks as symlinks (HF blobs/snapshots use
    # them), preserve perms+times; drop owner/group so it works over a
    # root-squashed NFS export. --partial resumes an interrupted copy.
    _RSYNC = ["rsync", "-rlt", "--partial", "--delete"]

    async def _run_copy(self, argv: list[str], dest: Path, total: int) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        while self._proc.returncode is None:
            if self._cancel:
                break
            self._touch(done_bytes=await _du_local(dest))
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                pass
        rc = await self._proc.wait()
        if self._cancel:
            raise asyncio.CancelledError()
        if rc != 0:
            err = (await self._proc.stderr.read()).decode(errors="replace")
            raise RuntimeError(err.strip()[-500:] or f"rsync exit {rc}")

    async def _backup(self, repo: str, base: str) -> None:
        try:
            dirname = hf.repo_dirname(repo)
            if not hf._DIRNAME_RE.match(dirname):
                raise ValueError("invalid model id")
            src = await _source_node(dirname)
            if src is None:
                raise RuntimeError(f"{repo} is not cached on any node")
            total = await hf._du_bytes(src, dirname)
            if total <= 0:
                raise RuntimeError("source model appears empty")
            dest = _root(base) / dirname
            await _mkdirp(dest)
            have = await _du_local(dest)
            need = int((total - have) * (1 + hf.SPACE_MARGIN))
            free = _free_bytes(str(dest))
            if need > 0 and free < need:
                raise RuntimeError(
                    f"not enough space on the backup target: needs "
                    f"~{need / 1e9:.1f} GB free, has {free / 1e9:.1f} GB")
            self._touch(total_bytes=total, phase="copying",
                        message=f"Backing up from {src['name']} → {dest}")

            if src["local"]:
                local_src = os.path.expanduser(
                    f"~/.cache/huggingface/hub/{dirname}/")
                argv = self._RSYNC + [local_src, str(dest) + "/"]
            else:
                cmd = " ".join(self._RSYNC) + \
                    f" ~/.cache/huggingface/hub/{dirname}/ " + \
                    shlex.quote(str(dest) + "/")
                argv = hf._ssh(src["ip"], cmd)

            await self._run_copy(argv, dest, total)

            recipes = await recipes_for_model(repo)
            final = await _du_local(dest)
            manifest = {
                "repo": repo, "dirname": dirname, "size_bytes": final,
                "recipes": recipes, "source_node": src["name"],
                "saved_at": time.time(),
            }
            await asyncio.to_thread(
                (dest / config.BACKUP_MANIFEST).write_text,
                json.dumps(manifest, indent=2))
            self._touch(active=False, phase="done", done_bytes=final,
                        recipes=recipes, message="Backup complete")
        except asyncio.CancelledError:
            self._touch(active=False, phase="cancelled", message="Cancelled")
        except Exception as exc:
            self._touch(active=False, phase="error",
                        error=f"{type(exc).__name__}: {exc}", message="Failed")

    async def _verify(self, repo: str, base: str) -> None:
        """Prove the backup is restorable: checksum-compare it against the
        cached copy (dry-run rsync reports any file that differs, is missing,
        or is extra). Falls back to a size-vs-manifest check when the model is
        no longer cached anywhere."""
        try:
            dirname = hf.repo_dirname(repo)
            if not hf._DIRNAME_RE.match(dirname):
                raise ValueError("invalid model id")
            bdir = _root(base) / dirname
            if not await asyncio.to_thread(bdir.is_dir):
                raise RuntimeError("no backup found at that location")

            src = await _source_node(dirname)
            if src is None:
                # Weak check only: nothing cached to checksum against.
                man = _read_manifest(bdir)
                expect, actual = man.get("size_bytes"), await _du_local(bdir)
                if not expect:
                    raise RuntimeError(
                        "model is not cached on any node and the backup has "
                        "no size manifest — nothing to verify against")
                if abs(actual - expect) > max(4096, expect * 0.001):
                    raise RuntimeError(
                        f"size mismatch vs manifest: backup is "
                        f"{actual / 1e9:.2f} GB, manifest says "
                        f"{expect / 1e9:.2f} GB")
                self._touch(active=False, phase="done",
                            message="Size matches the manifest (model not "
                                    "cached, so checksums were not compared)")
                return

            self._touch(phase="verifying",
                        message=f"Checksumming against {src['name']}'s cache…")
            flags = ("rsync -rltn --checksum --itemize-changes --delete "
                     f"--exclude {config.BACKUP_MANIFEST}")
            if src["local"]:
                cache_src = os.path.expanduser(
                    f"~/.cache/huggingface/hub/{dirname}/")
                argv = flags.split() + [cache_src, str(bdir) + "/"]
            else:
                # Worker: the share is mounted at the same path cluster-wide.
                cmd = f"{flags} ~/.cache/huggingface/hub/{dirname}/ " + \
                    shlex.quote(str(bdir) + "/")
                argv = hf._ssh(src["ip"], cmd)

            t0 = time.time()
            self._proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            comm = asyncio.ensure_future(self._proc.communicate())
            while not comm.done():
                self._touch(message=f"Checksumming against {src['name']}'s "
                                    f"cache… ({int(time.time() - t0)}s)")
                await asyncio.wait([comm], timeout=2.0)
            out, err = comm.result()
            if self._cancel:
                raise asyncio.CancelledError()
            if self._proc.returncode != 0:
                raise RuntimeError(
                    err.decode(errors="replace").strip()[-500:]
                    or f"rsync exit {self._proc.returncode}")

            # Itemized lines starting with "." are attribute-only differences
            # (e.g. a timestamp) — the content is identical, so not drift.
            drift = [l for l in out.decode(errors="replace").splitlines()
                     if l.strip() and not l.startswith(".")]
            if drift:
                sample = "; ".join(d.strip() for d in drift[:5])
                raise RuntimeError(
                    f"backup differs from {src['name']}'s cache in "
                    f"{len(drift)} entr{'y' if len(drift) == 1 else 'ies'}: "
                    f"{sample}")
            self._touch(active=False, phase="done",
                        message=f"Verified OK — checksums match "
                                f"{src['name']}'s cache "
                                f"({int(time.time() - t0)}s)")
        except asyncio.CancelledError:
            self._touch(active=False, phase="cancelled", message="Cancelled")
        except Exception as exc:
            self._touch(active=False, phase="error",
                        error=f"{type(exc).__name__}: {exc}", message="Failed")

    async def _restore(self, repo: str, base: str, mirror: bool) -> None:
        try:
            dirname = hf.repo_dirname(repo)
            if not hf._DIRNAME_RE.match(dirname):
                raise ValueError("invalid model id")
            if (await current_model_id()) == repo:
                raise RuntimeError(
                    "that model is currently loaded — stop the recipe first")
            srcdir = _root(base) / dirname
            if not await asyncio.to_thread(srcdir.is_dir):
                raise RuntimeError("no backup found at that location")
            total = await _du_local(srcdir)
            dest = hf.HF_HUB / dirname
            await _mkdirp(dest)
            have = await _du_local(dest)
            need = int((total - have) * (1 + hf.SPACE_MARGIN))
            free = _free_bytes(str(dest))
            if need > 0 and free < need:
                raise RuntimeError(
                    f"not enough space on {_HEAD['name']}: needs "
                    f"~{need / 1e9:.1f} GB free, has {free / 1e9:.1f} GB")
            self._touch(total_bytes=total, phase="copying",
                        message=f"Restoring to {_HEAD['name']}…")

            argv = self._RSYNC + [
                "--exclude", config.BACKUP_MANIFEST,
                str(srcdir) + "/", str(dest) + "/"]
            await self._run_copy(argv, dest, total)

            do_mirror = mirror and bool(_WORKERS)
            if do_mirror:
                self._touch(active=False, phase="done",
                            done_bytes=await _du_local(dest), mirroring=True,
                            message="Restored to head — mirroring to cluster…")
                try:
                    hf.preloader.start(repo, mirror=True, existing=True)
                except RuntimeError as exc:
                    self._touch(mirroring=False,
                                message=f"Restored to head. Mirror skipped: {exc}")
            else:
                self._touch(active=False, phase="done",
                            done_bytes=await _du_local(dest),
                            message="Restore complete")
        except asyncio.CancelledError:
            self._touch(active=False, phase="cancelled", message="Cancelled")
        except Exception as exc:
            self._touch(active=False, phase="error",
                        error=f"{type(exc).__name__}: {exc}", message="Failed")


manager = BackupManager()
