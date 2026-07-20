# SparkDash

A monitoring **and control** dashboard for a [sparkrun](https://sparkrun.dev)
DGX Spark cluster. It reports the Ray cluster, the vLLM instance, the running
recipe / loaded model, and per-node vitals (CPU, RAM, VRAM, GPU, storage); and,
behind admin auth over HTTPS, it preloads models, manages the TLS certificate,
streams live vLLM logs, and starts / stops / restarts recipes. Runs on a single
Spark or a multi-node cluster. Reads stay public; writes require login.

## Run

Must run on the **head node** — the Ray dashboard binds to `127.0.0.1`.
Serves **HTTPS**; a self-signed cert is generated on first run.

First, configure your cluster (copy `sparkdash.example.toml` to
`~/.config/sparkdash/config.toml` and fill in your nodes — see
[Configuration](#configuration)). Then:

```bash
./run.sh                 # https://<head-host>:7862
SPARKDASH_PORT=8080 ./run.sh
```

Open `https://<head-host>:7862` from any machine on the LAN (the browser
will warn about the self-signed cert until you install your own — see below).

### Set the admin password

Write actions require login. Set the password once (username defaults to
`sparkadmin`):

```bash
python -m sparkdash.admin set-password
# or non-interactively:
SPARKDASH_ADMIN_PASSWORD=... python -m sparkdash.admin set-password
```

### Install as a service (starts on boot)

The installer deploys a self-contained copy to **`/opt/sparkdash`** (separate
from your dev checkout), builds an isolated venv there from the lockfile, then
installs, **enables, and starts** the systemd unit:

```bash
./deploy/install.sh                        # -> /opt/sparkdash
SPARKDASH_PREFIX=/srv/sparkdash ./deploy/install.sh   # custom location
```

It's **idempotent** — re-run it to deploy changes (it re-syncs the code,
re-locks the venv, and restarts the service). Runtime state
(`~/.local/share/sparkdash`: DB + certs) is never touched, so the admin
password and certificate survive updates.

```bash
systemctl status sparkdash        # state
journalctl -u sparkdash -f        # follow logs
sudo systemctl restart sparkdash  # manual restart
```

The unit (`deploy/sparkdash.service`) runs `/opt/sparkdash/.venv/bin/python -m
sparkdash` as the `nvidia` user and sets `HOME` + `PATH` so the SSH VRAM probe
and `sparkrun` subprocesses work at boot. `run.sh` still launches straight from
the dev checkout for iteration.

## Configuration

Cluster topology (node IPs, SSH user) lives in a TOML file — no addresses are
baked into the source. It's searched in order: `$SPARKDASH_CONFIG`,
`~/.config/sparkdash/config.toml`, then `<repo>/sparkdash.toml`. With no config
file, SparkDash runs single-node against localhost.

Copy the template and edit it:

```bash
cp sparkdash.example.toml ~/.config/sparkdash/config.toml
$EDITOR ~/.config/sparkdash/config.toml
```

```toml
ssh_user = "nvidia"            # SSH user shared across nodes
sparkrun_cluster = "default"   # sparkrun saved-cluster name

[[nodes]]                      # one block per machine
name  = "head-node"
ip    = "head.example"         # management address (host or IP) — MUST match
                               #   the host key `sparkrun cluster monitor` reports
rdma  = "head-fast.example"    # fast-copy address for model mirroring
                               #   (set equal to ip if there's no separate link)
role  = "head"                 # "head" or "worker" (label only)
local = true                   # true ONLY on the node SparkDash runs on

[[nodes]]
name  = "worker-node"
ip    = "worker.example"
rdma  = "worker-fast.example"
role  = "worker"
local = false

[cert]                         # extra SANs for the self-signed TLS cert
hostnames = ["localhost"]
ips       = ["127.0.0.1"]
```

Notes:
- **`ip` must match the monitor stream** — SparkDash maps `sparkrun cluster
  monitor` output back to nodes by this address; a mismatch leaves that node's
  tiles empty.
- **`local = true`** marks the one node where the backend runs (probes run
  directly there instead of over SSH). Set it on exactly one node.
- **`rdma`** is used only for the model-mirror copy; Ray/vLLM/probes use `ip`.
- **Add a node** by appending another `[[nodes]]` block (`local = false`) — the
  dashboard, VRAM probe, and model mirror all pick it up, provided passwordless
  SSH from the head reaches it as `ssh_user`.

**Apply changes** by restarting the service (the file is read once at startup,
and lives outside the install dir so `install.sh` never touches it):

```bash
sudo systemctl restart sparkdash
```

## How it works

A single FastAPI app fans out to every data source, merges the results into one
snapshot, and broadcasts it to the browser over a WebSocket every 2s.

| Source | What it feeds |
|--------|---------------|
| `sparkrun cluster monitor --json` (persistent subprocess) | per-node CPU / RAM / GPU util / temp / power, 1s cadence |
| SSH node probe (`nvidia-smi --query-compute-apps` + `df`) | per-process **VRAM** (only path that works on GB10 unified memory) + disk |
| Ray dashboard `127.0.0.1:8265` | cluster health, node liveness, resources |
| vLLM `:8000` `/health` `/v1/models` `/metrics` | health, loaded model, serving metrics |
| `sparkrun status` | running recipe / job + containers |

Control traffic uses each node's management address; the optional per-node
`rdma` address (a fast direct-connect link) is reserved for bulk file moves
(model mirroring). Both are set per node in the config file.

## Consuming the data elsewhere

Collection is decoupled from the bundled UI, so other dashboards can pull the
same data. The endpoints below are read-only and side-effect free (the write
actions live under `/api/admin/*` and require login).

| Endpoint | Format | For |
|----------|--------|-----|
| `GET /api/snapshot` | JSON (full merged state) | custom apps; tracks latest schema |
| `GET /api/v1/snapshot` | JSON | same payload, pinned schema version |
| `GET /metrics` | Prometheus exposition | Prometheus / Grafana scrape target |
| `WS /ws` | JSON push every 2s | live browser clients |

### Prometheus

`/metrics` re-exports the snapshot with a `sparkdash_` prefix — including
`sparkdash_node_vram_used_bytes`, the GB10 per-process VRAM that neither vLLM's
nor Ray's own metrics expose. Node series are labelled `node` and `role`.

```yaml
scrape_configs:
  - job_name: sparkdash
    scheme: https
    tls_config:
      insecure_skip_verify: true   # self-signed; or point ca_file at your cert
    static_configs:
      - targets: ["<head-host>:7862"]
```

Key series: `sparkdash_cluster_healthy`, `sparkdash_ray_nodes_alive`,
`sparkdash_vllm_healthy`, `sparkdash_vllm_info{model}`,
`sparkdash_recipe_info{name,tp}`, and per-node
`sparkdash_node_{cpu_percent,memory_used_bytes,vram_used_bytes,gpu_*,disk_*}`.

### JSON snapshot shape

```
{ ts, cluster_healthy,
  ray:    { reachable, nodes_alive, nodes_total, nodes: [ {node_ip, state, is_head, cpu, gpu, memory_bytes, object_store_bytes} ] },
  vllm:   { reachable, healthy, model, max_model_len, metrics: { "vllm:...": float, prefix_cache_hit_rate } },
  recipe: { running, name, tp, id, containers: [ {role, ip, status} ] },
  nodes:  [ { name, ip, role, hostname, online,
              cpu_pct, cpu_load_1m, cpu_temp_c, cpu_freq_mhz,
              mem_used_mb, mem_total_mb, mem_used_pct, swap_used_mb, swap_total_mb,
              gpu_name, gpu_util_pct, gpu_temp_c, gpu_power_w, gpu_clock_mhz,
              vram_used_mb, disk_used, disk_total, uptime_sec } ] }
```

Memory/VRAM are MiB; disk is bytes. Any field may be `null` if that probe is
momentarily unavailable — consumers should treat missing values as unknown, not
zero (the `/metrics` exporter omits such series for exactly this reason).

## Authentication & TLS

Reads stay public; **write actions are gated**. Two credential types share one
gate, so automation and humans are handled differently:

| Credential | How | Use for |
|------------|-----|---------|
| **Session** | password login (`sparkadmin`) → `httponly` cookie, 12h | interactive admin; **required** to mint tokens or replace the cert |
| **API token** | minted in the UI, sent as `Authorization: Bearer …` | programmatic write access (scripts/CI) |

The split is deliberate: an API token can drive *operational* writes but
**cannot** mint tokens, revoke tokens, or replace the cert — that
*administration* requires an interactive session, so a leaked token can't
escalate into managing the system. Passwords are scrypt-hashed; tokens are
stored only as SHA-256 digests (shown once at creation). State lives in
`~/.local/share/sparkdash/sparkdash.db`, outside the repo.

### Certificate

HTTPS is served from `~/.local/share/sparkdash/certs/`. On first run a
self-signed cert (covering the node's hostnames/IPs) is generated. In the
**Admin → TLS Certificate** panel you can paste your own PEM cert + key; it's
validated (key matches cert, not expired, loads in OpenSSL) before install,
then the server auto-restarts (~2s) to apply it.

**Anti-lockout:** startup validates the configured cert and, if it's
missing/broken/expired, backs it up and regenerates a self-signed one — a bad
cert can never brick access. This is why the systemd unit uses `Restart=always`.

> On plain HTTP this would all cross the wire in cleartext, which is why the app
> serves everything over HTTPS. On the LAN the self-signed cert is fine; install
> a cert from your own CA to silence browser warnings.

## Layout

```
sparkdash/__main__.py    entrypoint: ensure TLS cert, then serve HTTPS
sparkdash.example.toml   cluster config template (copy & edit)
sparkdash/config.py      loads config; ports, paths, auth settings
sparkdash/collectors.py  Hub: background collectors + merged snapshot
sparkdash/app.py         FastAPI app, WebSocket broadcast, static serving
sparkdash/admin_api.py   auth / API-token / certificate routes
sparkdash/auth.py        password, sessions, tokens, require_admin/session gates
sparkdash/store.py       SQLite: admin credential, sessions, tokens
sparkdash/certs.py       self-signed generation, validation, anti-lockout
sparkdash/hf.py          model preload + RDMA mirror (reuses sparkrun's code)
sparkdash/backup.py      model backup/restore to a shared location (rsync)
sparkdash/recipe_ops.py  recipe start/stop/restart (reuses sparkrun's tooling)
sparkdash/logstream.py   live vLLM container-log streaming
sparkdash/chat.py        streaming chat proxy to the running model
sparkdash/admin.py       `set-password` CLI
frontend/index.html      dashboard page (read-only)
frontend/admin.html      full-page admin (login, models, tokens, cert)
frontend/app.css         shared stylesheet
deploy/install.sh        deploy to /opt/sparkdash + enable the systemd service
deploy/sparkdash.service systemd unit (installed path, Restart=always)
```

## Model preload

**Admin → Models** preloads a recipe's model so it's cached before you run the
recipe. Pick a recipe (or type any HF model id, incl. the `org/repo:QUANT` GGUF
form), and it downloads onto the head node and — with *Mirror to cluster* — copies
to the other node(s) over the **200GbE RDMA** link. Progress is shown per node.

It reuses sparkrun's own `download_model` / `distribute_model_from_head`
(invoked with sparkrun's interpreter), passing the RDMA IPs as
`worker_transfer_hosts`, so staging behaves exactly like a real recipe run.
Endpoints (all `require_admin`): `GET /api/admin/recipes`, `GET /api/admin/cache`,
`POST /api/admin/preload`, `GET /api/admin/preload/status`,
`POST /api/admin/preload/cancel`.

## Model backup & restore

**Admin → Models → Backup & Restore** saves a cached model to a shared location
so a 150 GB+ download never has to be repeated. Because the Mirror feature keeps
every node's cache byte-identical, a backup stores **one canonical copy** per
model — not one per host — under `<target>/Sparkdash/Models/`, keeping the
HuggingFace `models--<org>--<repo>` directory name so a restore is a plain copy
back into each node's cache.

Set the **backup target** (a base path such as an NFS mount, e.g. `/mnt/llm`;
remembered across sessions), then **Backup** any cached model. Each backup is
self-describing: a `sparkdash-backup.json` manifest records the repo, size,
source node, timestamp, and **which recipe(s) use the model**. Available backups
are listed with their recipe and a *cached* badge when the model is already
present locally.

**Restore** copies the backup onto the head node, then re-mirrors to the
worker(s) over RDMA (reusing the Mirror path). Restoring the currently-loaded
model is refused (stop the recipe first). Copies use `rsync` (symlink- and
resume-aware); one backup, restore, or verify runs at a time.

**Verify** proves a backup is restorable: a checksum comparison (dry-run
`rsync --checksum`) against the cached copy reports any file that differs, is
missing, or is extra; if the model is no longer cached anywhere it falls back
to a size-vs-manifest check. **Delete** removes a backup from the share (the
node caches are untouched), and the card shows the share's free space.

Every big copy — preload, mirror, backup, restore — is **guarded by a
disk-space preflight**: it refuses to start (with a clear message) unless the
target has room for the remainder of the model plus a small margin, rather than
failing 80 GB into a copy.

Endpoints (all `require_admin`): `GET/POST /api/admin/backup/target`,
`GET /api/admin/backups`, `POST /api/admin/backup`, `POST /api/admin/restore`,
`POST /api/admin/backup/verify`, `POST /api/admin/backup/delete`,
`GET /api/admin/backup/status`, `POST /api/admin/backup/cancel`.

## Recipe control

**Admin → Recipe Control** starts, stops, and restarts recipes, wrapping
sparkrun's own tooling. Topology is gated by the recipe's `min_nodes`: pick
**single node** (`--solo`) or **cluster** (`--cluster`); recipes needing ≥2
nodes force cluster. A dry-run **VRAM pre-flight** shows whether the model fits
before you commit. Start is blocked while a recipe runs (stop first); one
operation at a time, with live streamed output. Restart captures the exact
running config (`export running-recipe`) so it reproduces precisely. All
endpoints are `require_admin`. Endpoints: `GET /api/admin/recipe/current`,
`POST …/recipe/{preflight,start,stop,restart}`, `GET …/recipe/op`.

## Features

- Live per-node vitals (CPU, RAM, VRAM, GPU, storage) + Ray / vLLM / recipe status,
  with rolling **sparklines** (GPU utilisation, generation throughput)
- External data APIs + a Prometheus `/metrics` exporter (incl. GB10 per-process VRAM)
- HTTPS with admin auth (password sessions + API tokens) and in-app TLS cert management
- A tabbed admin page: **Recipes · Models · Chat · Settings**
- **Chat playground** against the running model (streams reasoning + answer)
- HuggingFace model preload with fast-network (RDMA) cluster mirroring, plus
  **cache management** (prune cached models to reclaim disk)
- **Model backup & restore** to a shared location (one canonical copy per model,
  with a recipe-aware manifest), so a huge download is never repeated — with
  checksum **verification**, backup retention (delete + free-space display),
  and **disk-space preflights** on every big copy
- Recipe start / stop / restart with single-node or cluster topology; stopped
  recipes are preserved and re-runnable
- Live vLLM log viewer
- Opt-in browser **notifications** (recipe ready, preload done, health changes)
- Single node or multi-node; no addresses baked into the source
