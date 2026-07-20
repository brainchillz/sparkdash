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
from typing import Any

import httpx

from . import config

# Remote one-shot probe: sum per-process VRAM (MiB) and read root fs used/total
# bytes. Emitted as "<vram_mib>|<used_bytes>,<total_bytes>".
_PROBE_CMD = (
    "V=$(nvidia-smi --query-compute-apps=used_gpu_memory "
    "--format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}'); "
    "D=$(df -B1 / | awk 'NR==2{print $3\",\"$2}'); echo \"$V|$D\""
)


class Hub:
    def __init__(self) -> None:
        # Latest raw pieces of state, updated independently by each task.
        self._monitor: dict[str, dict] = {}      # node name -> monitor frame
        self._probe: dict[str, dict] = {}         # node name -> {vram_mib,disk_used,disk_total}
        self._ray: dict[str, Any] = {"reachable": False}
        self._vllm: dict[str, Any] = {"reachable": False}
        self._recipe: dict[str, Any] = {"running": False}
        self._client = httpx.AsyncClient(timeout=4.0)
        self._tasks: list[asyncio.Task] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._run_monitor_stream()),
            asyncio.create_task(self._loop(self._poll_probes, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_ray, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_vllm, config.SLOW_POLL)),
            asyncio.create_task(self._loop(self._poll_status, config.STATUS_POLL)),
        ]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._client.aclose()

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

    # -- monitor stream (backbone) ----------------------------------------

    async def _run_monitor_stream(self) -> None:
        """Consume `sparkrun cluster monitor --json` as an NDJSON stream.

        Restarts the subprocess if it exits (e.g. transient SSH hiccup).
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
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for ip, data in frame.get("hosts", {}).items():
                        name = config.IP_TO_NODE.get(ip, ip)
                        self._monitor[name] = data
            except asyncio.CancelledError:
                if proc and proc.returncode is None:
                    proc.terminate()
                raise
            except Exception as exc:
                self._note_error("monitor_stream", exc)
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
        # Expected: "<vram_mib>|<used_bytes>,<total_bytes>"
        try:
            vram_part, disk_part = text.split("|", 1)
            used, total = disk_part.split(",", 1)
            self._probe[node["name"]] = {
                "vram_used_mb": float(vram_part or 0),
                "disk_used": int(used or 0),
                "disk_total": int(total or 0),
            }
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
            state["reachable"] = True
            state["healthy"] = h.status_code == 200
        except Exception:
            self._vllm = state
            return

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

    # -- merged snapshot ---------------------------------------------------

    def snapshot(self) -> dict:
        nodes = []
        for n in config.NODES:
            mon = self._monitor.get(n["name"], {})
            probe = self._probe.get(n["name"], {})
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
        return {
            "ts": time.time(),
            "cluster_healthy": bool(cluster_healthy),
            "node_count": len(nodes),
            "ray": self._ray,
            "vllm": self._vllm,
            "recipe": self._recipe,
            "nodes": nodes,
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
_METRIC_RE = re.compile(r"^(vllm:[a-zA-Z_]+)(?:\{[^}]*\})?\s+([0-9eE.+\-]+)$")

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


def _parse_vllm_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or "_created" in line:
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if name in _WANTED:
            try:
                out[name] = out.get(name, 0.0) + float(value)
            except ValueError:
                pass
    # Derived: prefix-cache hit rate.
    q = out.get("vllm:prefix_cache_queries_total", 0.0)
    h = out.get("vllm:prefix_cache_hits_total", 0.0)
    if q > 0:
        out["prefix_cache_hit_rate"] = h / q
    return out


# Header line, e.g.:  "Job: minimax-2.7  (tp=2)  [e6b6dfeb53aa]  (2 container(s))"
_JOB_RE = re.compile(
    r"Job:\s+(?P<name>\S+)\s+\(tp=(?P<tp>\d+)\)\s+\[(?P<id>[0-9a-f]+)\]"
)
# Container line, e.g.: "  head  <host>  Up 5 days  vllm-node-xxxxx"
_CONT_RE = re.compile(
    r"^\s+(head|worker)\s+(?P<ip>\d+\.\d+\.\d+\.\d+)\s+(?P<status>Up[^\n]*?)\s{2,}\S+\s*$"
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
        "id": m.group("id"),
        "containers": containers,
    }
