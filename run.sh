#!/usr/bin/env bash
# Launch SparkDash. Run on the head node — the Ray dashboard is
# bound to 127.0.0.1, so the backend must live there. Serves HTTPS; a
# self-signed cert is generated on first run.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m sparkdash "$@"
