"""Data collectors for SparkDash Phase 1 (read-only).

A single `Hub` owns all live state and runs background tasks that keep it fresh:

  * MonitorStream  - persistent `sparkrun cluster monitor --json` subprocess,
                     the high-frequency (1s) per-node CPU/RAM/GPU backbone.
  * node probe     - VRAM (per-process, the only path that works on GB10's
                     unified memory) + root-disk usage, per node.
  * Ray poll       - cluster health + node liveness from the head dashboard.
  * vLLM poll      - health, loaded model, and Prometheus serving metrics.
  * status poll    - `sparkrun status` for the running recipe / job.

The Hub merges everything into one snapshot dict that the app broadcasts.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from typing import Any

import httpx

from . import config

# Remote one-shot probe: sum per-process VRAM (MiB), read root fs used/total
# bytes, and list sparkrun containers (for orphan detection — a container
# sparkrun no longer tracks still squats the port and unified memory).
# Emitted as "<vram_mib>|<used_bytes>,<total_bytes>|<name,name,...>".
_PROBE_CMD = (
    "V=$(nvidia-smi --query-compute-apps=used_gpu_memory "
    "--format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}'); "
    "D=$(df -B1 / | awk 'NR==2{print $3\",\"$2}'); "
    "C=$(docker ps --format '{{.Names}}' 2>/dev/null | grep '^sparkrun_' "
    "| paste -sd, -); echo \"$V|$D|$C\""
)


class Hub:
    def __init__(self) -> None:
        # Latest raw pieces of state, updated independently by each task.
        self._monitor: dict[str, dict] = {}      # node name -> monitor frame
        self._monitor_ts: dict[str, float] = {}  # node name -> monotonic frame time
        self._probe: dict[str, dict] = {}         # node name -> {vram_mib,disk_used,disk_total}
        self._probe_ts: dict[str, float] = {}     # node name -> monotonic probe time
        self._ray: dict[str, Any] = {"reachable": False}
        self._vllm: dict[str, Any] = {"reachable": False}
        self._vllm_fails = 0
        self._recipe: dict[str, Any] = {"running": False}
        self._client = httpx.AsyncClient(timeout=4.0)
        self._tasks: list[asyncio.Task] = []
        # Recent collector failures, surfaced in the snapshot so a broken
        # feed is visible on the page instead of only in the journal.
        self._errors: deque[dict] = deque(maxlen=50)
        self._sparkrun_version = ""
        self._canary: dict[str, Any] = {}        # last canary result (see canary())
        self._peers: dict[str, dict] = {}        # peer name -> condensed snapshot

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._run_monitor_stream()),
            asyncio.create_task(self._loop(self._poll_probes, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_ray, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_vllm, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_status, config.STATUS_POLL)),
            # sparkrun version changes only on `sparkrun update`, but that can
            # happen underneath a running dashboard — refresh occasionally.
            asyncio.create_task(self._loop(self._poll_version, 600.0)),
        ]
        if config.PEERS:
            self._tasks.append(asyncio.create_task(
                self._loop(self._poll_peers, config.PEER_POLL)))
        if config.CANARY_INTERVAL > 0:
            self._tasks.append(asyncio.create_task(
                self._loop(self.run_canary, config.CANARY_INTERVAL)))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._client.aclose()
        if hasattr(self, "_peer_client"):
            await self._peer_client.aclose()

    async def _loop(self, fn, interval: float) -> None:
        """Run an async collector forever, swallowing per-cycle errors."""
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the loop alive on transient failures
                self._note_error(fn.__name__, exc)
            await asyncio.sleep(interval)

    def _note_error(self, where: str, exc: Exception) -> None:
        print(f"[sparkdash] {where}: {type(exc).__name__}: {exc}", flush=True)
        self._errors.append({"ts": time.time(), "where": where,
                             "error": f"{type(exc).__name__}: {exc}"})

    def _failing_collectors(self, window: float = 120.0, threshold: int = 3) -> list[str]:
        """Collectors with repeated recent failures — a feed that is down, not
        a one-off hiccup. Returns human-readable summaries."""
        cutoff = time.time() - window
        counts: dict[str, dict] = {}
        for e in self._errors:
            if e["ts"] >= cutoff:
                c = counts.setdefault(e["where"], {"n": 0, "last": ""})
                c["n"] += 1
                c["last"] = e["error"]
        return [f"{where}: failing repeatedly ({c['n']}x in {int(window)}s — {c['last']})"
                for where, c in sorted(counts.items()) if c["n"] >= threshold]

    # -- monitor stream (backbone) ----------------------------------------

    async def _run_monitor_stream(self) -> None:
        """Consume `sparkrun cluster monitor --json` as an NDJSON stream.

        Restarts the subprocess if it exits (e.g. transient SSH hiccup) or
        wedges. sparkrun never retries a host connection that hangs before
        its first sample — it streams `{"connecting": true}` placeholder
        frames forever (seen after a boot where the nodes' sshd wasn't up
        yet when the stream started) — so only frames carrying real host
        data reset the staleness clock, and a stream that goes
        MONITOR_STALE without one is killed and respawned.
        """
        while True:
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "sparkrun", "cluster", "monitor",
                    "--cluster", config.SPARKRUN_CLUSTER,
                    "--json", "--interval", "1",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                assert proc.stdout is not None
                last_data = time.monotonic()
                while True:
                    raw = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=config.MONITOR_STALE)
                    if not raw:      # EOF: subprocess exited
                        break
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    now = time.monotonic()
                    hosts = frame.get("hosts", {})
                    # sparkrun <= 0.2.x keys "hosts" by IP; 0.3.x emits a
                    # list of {"host": ip, "sample": {...}, ...} entries.
                    if isinstance(hosts, list):
                        pairs = [
                            (h.get("host"), h.get("sample"))
                            for h in hosts if isinstance(h, dict)
                        ]
                    else:
                        pairs = hosts.items()
                    for ip, data in pairs:
                        # Placeholder frames ({"connecting": true} /
                        # {"error": ...}) carry no metrics; storing them
                        # would render as an online node with blank stats.
                        if not ip or not isinstance(data, dict) \
                                or "hostname" not in data:
                            continue
                        name = config.IP_TO_NODE.get(ip, ip)
                        self._monitor[name] = data
                        self._monitor_ts[name] = now
                        last_data = now
                    if time.monotonic() - last_data > config.MONITOR_STALE:
                        raise TimeoutError(
                            "no host data for %.0fs (stream wedged)"
                            % config.MONITOR_STALE)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._note_error("monitor_stream", exc)
            finally:
                if proc and proc.returncode is None:
                    proc.terminate()
            # Stream ended or errored; pause and respawn.
            await asyncio.sleep(3.0)

    # -- node probe: VRAM + disk ------------------------------------------

    async def _poll_probes(self) -> None:
        await asyncio.gather(*(self._probe_node(n) for n in config.NODES))

    async def _probe_node(self, node: dict) -> None:
        if node["local"]:
            argv = ["bash", "-c", _PROBE_CMD]
        else:
            argv = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                f"{config.SSH_USER}@{node['ip']}", _PROBE_CMD,
            ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        text = out.decode(errors="replace").strip()
        # Expected: "<vram_mib>|<used_bytes>,<total_bytes>|<container,...>"
        # (the container field is absent from pre-update probes — tolerate it).
        try:
            vram_part, rest = text.split("|", 1)
            disk_part, _, cont_part = rest.partition("|")
            used, total = disk_part.split(",", 1)
            self._probe[node["name"]] = {
                "vram_used_mb": float(vram_part or 0),
                "disk_used": int(used or 0),
                "disk_total": int(total or 0),
                "containers": [c for c in cont_part.split(",") if c],
            }
            self._probe_ts[node["name"]] = time.monotonic()
        except ValueError:
            self._note_error("probe_parse", ValueError(f"bad probe output: {text!r}"))

    # -- Ray dashboard -----------------------------------------------------

    async def _poll_ray(self) -> None:
        try:
            r = await self._client.get(f"{config.RAY_DASHBOARD}/api/v0/nodes?detail=true")
            r.raise_for_status()
            payload = r.json()
        except Exception:
            self._ray = {"reachable": False}
            return

        nodes = []
        alive = 0
        for n in payload.get("data", {}).get("result", {}).get("result", []):
            is_alive = n.get("state") == "ALIVE"
            alive += 1 if is_alive else 0
            res = n.get("resources_total", {})
            nodes.append({
                "node_ip": n.get("node_ip"),
                "state": n.get("state"),
                "is_head": n.get("is_head_node", False),
                "cpu": res.get("CPU"),
                "gpu": res.get("GPU"),
                "memory_bytes": res.get("memory"),
                "object_store_bytes": res.get("object_store_memory"),
            })
        self._ray = {
            "reachable": True,
            "nodes_alive": alive,
            "nodes_total": len(nodes),
            "nodes": nodes,
        }

    # -- vLLM --------------------------------------------------------------

    async def _poll_vllm(self) -> None:
        state: dict[str, Any] = {"reachable": False, "healthy": False}
        try:
            h = await self._client.get(f"{config.VLLM_BASE}/health")
        except Exception as exc:
            # One dropped request must not flap the UI red: keep the last good
            # state until a second consecutive cycle also fails. Log only while
            # we still believed vLLM was up, so a steady-state outage doesn't
            # spam the journal every cycle.
            self._vllm_fails += 1
            if self._vllm.get("reachable"):
                self._note_error("poll_vllm", exc)
            if self._vllm_fails >= 2 or not self._vllm.get("reachable"):
                self._vllm = state
            return
        self._vllm_fails = 0
        state["reachable"] = True
        state["healthy"] = h.status_code == 200

        try:
            m = await self._client.get(f"{config.VLLM_BASE}/v1/models")
            data = m.json().get("data", [])
            if data:
                model = data[0]
                state["model"] = model.get("id")
                state["max_model_len"] = model.get("max_model_len")
        except Exception:
            pass

        try:
            met = await self._client.get(f"{config.VLLM_BASE}/metrics")
            state["metrics"] = _parse_vllm_metrics(met.text)
        except Exception:
            state["metrics"] = {}

        self._vllm = state

    # -- sparkrun status / recipe -----------------------------------------

    async def _poll_status(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "sparkrun", "status", "--cluster", config.SPARKRUN_CLUSTER,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
        self._recipe = _parse_status(out.decode(errors="replace"))

    # -- sparkrun version --------------------------------------------------

    async def _poll_version(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "sparkrun", "--version",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        m = re.search(r"version\s+(\S+)", out.decode(errors="replace"))
        if m:
            self._sparkrun_version = m.group(1)

    # -- peer installs -----------------------------------------------------

    async def _poll_peers(self) -> None:
        # Peers serve HTTPS with self-signed certs; nothing secret is read
        # from them (the snapshot endpoint is public), so skip verification.
        if not hasattr(self, "_peer_client"):
            self._peer_client = httpx.AsyncClient(timeout=4.0, verify=False)

        async def one(peer: dict) -> None:
            try:
                r = await self._peer_client.get(peer["url"] + "/api/v1/snapshot")
                r.raise_for_status()
                s = r.json()
                self._peers[peer["name"]] = {
                    "name": peer["name"], "url": peer["url"],
                    "reachable": True,
                    "healthy": bool(s.get("cluster_healthy")),
                    "model": (s.get("vllm") or {}).get("model"),
                    "node_count": s.get("node_count"),
                    "ts": time.time(),
                }
            except Exception:
                self._peers[peer["name"]] = {
                    "name": peer["name"], "url": peer["url"],
                    "reachable": False, "ts": time.time(),
                }
        await asyncio.gather(*(one(p) for p in config.PEERS))

    # -- model canary ------------------------------------------------------

    async def run_canary(self) -> dict:
        """One real (tiny) chat completion: proves coherent output and
        measures TTFT + decode rate. `/health` can't do either — vLLM binds
        the port before loading, and a garbling model still serves 200s."""
        result: dict[str, Any] = {"ts": time.time(), "ok": False}
        model = self._vllm.get("model")
        if not model:
            result["error"] = "no model loaded"
            self._canary = result
            return result
        body = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Reply with exactly: CANARY OK"}],
            "max_tokens": config.CANARY_MAX_TOKENS,
            "stream": True,
            # Both spellings of "don't think": DeepSeek templates use
            # `thinking`, Qwen templates use `enable_thinking`. Jinja ignores
            # whichever one a template doesn't know.
            "chat_template_kwargs": {"thinking": False,
                                     "enable_thinking": False},
        }
        text, t0, t_first = "", time.monotonic(), None
        try:
            async with self._client.stream(
                    "POST", f"{config.VLLM_BASE}/v1/chat/completions",
                    json=body, timeout=60.0) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        delta = (json.loads(line[6:])["choices"][0]
                                 .get("delta", {}).get("content") or "")
                    except (json.JSONDecodeError, LookupError):
                        continue
                    if delta and t_first is None:
                        t_first = time.monotonic()
                    text += delta
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            self._canary = result
            return result
        wall = time.monotonic() - t0
        result.update({
            "model": model,
            "reply": text.strip()[:120],
            "coherent": "CANARY OK" in text,
            "ttft_s": round(t_first - t0, 3) if t_first else None,
            "wall_s": round(wall, 2),
        })
        result["ok"] = bool(result["coherent"])
        if not result["ok"]:
            result["error"] = "reply did not contain the expected text"
        self._canary = result
        return result

    # -- merged snapshot ---------------------------------------------------

    def snapshot(self) -> dict:
        nodes = []
        now = time.monotonic()
        for n in config.NODES:
            mon = self._monitor.get(n["name"], {})
            # Expire monitor data the stream hasn't refreshed: presenting a
            # node as online with old numbers is worse than showing it down.
            if now - self._monitor_ts.get(n["name"], 0.0) > config.MONITOR_STALE:
                mon = {}
            probe = self._probe.get(n["name"], {})
            if now - self._probe_ts.get(n["name"], 0.0) > config.PROBE_STALE:
                probe = {}
            # Containers on this node that don't belong to the current job:
            # old-version leftovers sparkrun no longer tracks, squatting the
            # port and unified memory.
            rid = self._recipe.get("id") or ""
            orphans = [c for c in probe.get("containers", [])
                       if not (rid and rid in c)]
            nodes.append({
                "name": n["name"],
                "ip": n["ip"],
                "role": n["role"],
                "hostname": mon.get("hostname"),
                "online": bool(mon),
                "cpu_pct": _f(mon.get("cpu_usage_pct")),
                "cpu_load_1m": _f(mon.get("cpu_load_1m")),
                "cpu_temp_c": _f(mon.get("cpu_temp_c")),
                "cpu_freq_mhz": _f(mon.get("cpu_freq_mhz")),
                "mem_used_mb": _f(mon.get("mem_used_mb")),
                "mem_total_mb": _f(mon.get("mem_total_mb")),
                "mem_used_pct": _f(mon.get("mem_used_pct")),
                "swap_used_mb": _f(mon.get("swap_used_mb")),
                "swap_total_mb": _f(mon.get("swap_total_mb")),
                "gpu_name": mon.get("gpu_name"),
                "gpu_util_pct": _f(mon.get("gpu_util_pct")),
                "gpu_temp_c": _f(mon.get("gpu_temp_c")),
                "gpu_power_w": _f(mon.get("gpu_power_w")),
                "gpu_clock_mhz": _f(mon.get("gpu_clock_mhz")),
                "uptime_sec": _f(mon.get("uptime_sec")),
                # From the probe (GB10 unified-memory workaround + disk).
                "vram_used_mb": probe.get("vram_used_mb"),
                "disk_used": probe.get("disk_used"),
                "disk_total": probe.get("disk_total"),
                "orphan_containers": orphans,
            })

        # Healthy = all configured nodes reporting and vLLM serving. Ray is only
        # required to be consistent *if it's present* — some single-node setups
        # don't run Ray, and its absence shouldn't read as "degraded".
        ray_ok = (not self._ray.get("reachable")) or (
            self._ray.get("nodes_alive") == self._ray.get("nodes_total"))
        cluster_healthy = (
            self._vllm.get("healthy")
            and all(nd["online"] for nd in nodes)
            and ray_ok
        )

        # WHY unhealthy — every component of the verdict, in plain words, so
        # "degraded" is never a puzzle that needs SSH and journalctl to solve.
        reasons: list[str] = []
        if not self._vllm.get("reachable"):
            reasons.append("vLLM API unreachable")
        elif not self._vllm.get("healthy"):
            reasons.append("vLLM reachable but /health failing")
        for nd in nodes:
            if not nd["online"]:
                age = now - self._monitor_ts.get(nd["name"], 0.0)
                reasons.append(
                    f"node {nd['name']}: no monitor data"
                    + (f" for {int(age)}s" if self._monitor_ts.get(nd["name"]) else " yet"))
        if not ray_ok:
            reasons.append(f"Ray: {self._ray.get('nodes_alive')}/"
                           f"{self._ray.get('nodes_total')} nodes alive")

        # Non-fatal but worth a banner: failing collectors, orphan containers,
        # a canary that last failed.
        warnings = self._failing_collectors()
        for nd in nodes:
            for c in nd["orphan_containers"]:
                warnings.append(f"node {nd['name']}: orphan container {c}")
        if self._canary and not self._canary.get("ok"):
            warnings.append("model canary failed: "
                            + str(self._canary.get("error", "incoherent reply")))

        return {
            "ts": time.time(),
            "cluster_healthy": bool(cluster_healthy),
            "health_reasons": reasons,
            "warnings": warnings,
            "node_count": len(nodes),
            "sparkrun_version": self._sparkrun_version,
            "ray": self._ray,
            "vllm": self._vllm,
            "recipe": self._recipe,
            "nodes": nodes,
            "canary": self._canary,
            "peers": sorted(self._peers.values(), key=lambda p: p["name"]),
            "errors_recent": list(self._errors)[-10:],
        }


# -- parsing helpers -------------------------------------------------------

def _f(val: Any) -> float | None:
    """Coerce monitor string fields (which may be '') to float or None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# metric-name -> value, ignoring labels. We only keep the last sample per name,
# which is correct for this single-engine deployment. `_created` timestamp
# series are skipped so they can't be mistaken for counters.
_METRIC_RE = re.compile(
    r"^((?:vllm|sglang):[a-zA-Z_]+)(?:\{[^}]*\})?\s+([0-9eE.+\-]+)$")

_WANTED = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
}

# SGLang serves the same concepts under its own prefix (run with
# --enable-metrics). Normalised to the vllm: names so every consumer —
# frontend, history sampler, Prometheus exporter — is engine-agnostic.
_SGLANG_MAP = {
    "sglang:num_running_reqs": "vllm:num_requests_running",
    "sglang:num_queue_reqs": "vllm:num_requests_waiting",
    "sglang:token_usage": "vllm:kv_cache_usage_perc",
    "sglang:prompt_tokens_total": "vllm:prompt_tokens_total",
    "sglang:generation_tokens_total": "vllm:generation_tokens_total",
    "sglang:num_requests_total": "vllm:request_success_total",
    "sglang:cache_hit_rate": "prefix_cache_hit_rate",   # already a 0-1 ratio
}


def _parse_vllm_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or "_created" in line:
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        name = _SGLANG_MAP.get(name, name)
        if name in _WANTED or name == "prefix_cache_hit_rate":
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    # Derived: prefix-cache hit rate (vLLM exposes the two counters instead).
    q = out.get("vllm:prefix_cache_queries_total", 0.0)
    h = out.get("vllm:prefix_cache_hits_total", 0.0)
    if q > 0:
        out["prefix_cache_hit_rate"] = h / q
    return out


# Header line. Cluster jobs: "Job: minimax-2.7  (tp=2)  [e6b6dfeb53aa]  (2 container(s))"
# Solo jobs (newer sparkrun): "Job: @official/foo-vllm  (tp=1, pp=1)  [d6b0...]  (1 container(s))"
# Newer sparkrun uses composite ids with an underscore, e.g. [50294067b4ab6802_462aafd285e6].
_JOB_RE = re.compile(
    r"Job:\s+(?P<name>\S+)\s+\(tp=(?P<tp>\d+)(?:,\s*pp=(?P<pp>\d+))?\)\s+\[(?P<id>[0-9a-f_]+)\]"
)
# Container line, e.g.: "  head  <host>  Up 5 days  vllm-node-xxxxx"
# Single-node jobs report role "solo".
# sparkrun 0.3.x cluster jobs print roles node_0/node_1/... instead of
# head/worker; missing them empties the container list, which downstream
# reads as a solo job — and a "solo" restart of a tp=2 recipe crashes vLLM.
_CONT_RE = re.compile(
    r"^\s+(head|worker|solo|node_\d+)\s+(?P<ip>\d+\.\d+\.\d+\.\d+)\s+(?P<status>Up[^\n]*?)\s{2,}\S+\s*$"
)


def _parse_status(text: str) -> dict:
    m = _JOB_RE.search(text)
    if not m:
        return {"running": False, "raw": text.strip()}
    containers = []
    for line in text.splitlines():
        cm = _CONT_RE.match(line)
        if cm:
            containers.append({
                "role": cm.group(1),
                "ip": cm.group("ip"),
                "status": cm.group("status").strip(),
            })
    # sparkrun names a job after the recipe argument, which is a file path when
    # run from a recipe file (e.g. a restart) — show a clean basename instead.
    name = m.group("name")
    if name.startswith("/") or name.endswith((".yaml", ".yml")):
        name = re.sub(r"\.ya?ml$", "", name.rsplit("/", 1)[-1])

    return {
        "running": True,
        "name": name,
        "tp": int(m.group("tp")),
        "pp": int(m.group("pp")) if m.group("pp") else None,
        "id": m.group("id"),
        "solo": any(c["role"] == "solo" for c in containers),
        "containers": containers,
    }
