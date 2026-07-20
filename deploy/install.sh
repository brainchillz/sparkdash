#!/usr/bin/env bash
# Install (or update) SparkDash into a dedicated directory and run it as a
# systemd service. Deploys from this checkout — safe to re-run to update.
#
#   ./deploy/install.sh
#   SPARKDASH_PREFIX=/srv/sparkdash ./deploy/install.sh
#
# Runtime state (SQLite DB + TLS certs) lives in ~/.local/share/sparkdash and is
# never touched by install/update, so the admin password and certs persist.
set -euo pipefail

PREFIX="${SPARKDASH_PREFIX:-/opt/sparkdash}"
SERVICE="sparkdash.service"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SVC_USER="$(id -un)"          # the service runs as whoever installs it
SVC_GROUP="$(id -gn)"

UV="$(command -v uv || true)"
[[ -n "$UV" ]] || { echo "error: 'uv' not found in PATH." >&2; exit 1; }

echo "SparkDash install"
echo "  source : $SRC"
echo "  target : $PREFIX"
echo "  service: $SERVICE (user: $SVC_USER)"
echo

# Reminder: install on the Ray head node — Ray's dashboard binds to 127.0.0.1
# there, so Ray/vLLM data is only available on the head.
echo "note: install on the Ray head node (Ray's dashboard is bound to localhost)."
echo

# 1. Install dir, owned by the service user so the rest needs no sudo.
sudo mkdir -p "$PREFIX"
sudo chown "$SVC_USER":"$SVC_GROUP" "$PREFIX"

# 2. Mirror the code in (never the venv, git, local state, or secrets).
rsync -a --delete \
  --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/' \
  --exclude '*.pyc' --exclude '*-token' --exclude '*.token' \
  --exclude '.python-version' --exclude 'certs/' --exclude 'data/' \
  --exclude '*.db' --exclude '*.db-wal' --exclude '*.db-shm' \
  "$SRC/" "$PREFIX/"

# 3. Build an isolated venv in the install dir from the lockfile.
( cd "$PREFIX" && "$UV" sync --frozen )

# 4. Install the unit, substituting PREFIX, then enable + (re)start it.
sudo sed "s#/opt/sparkdash#$PREFIX#g" "$PREFIX/deploy/$SERVICE" \
  | sudo tee "/etc/systemd/system/$SERVICE" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"     # restart so updates take effect

echo
sudo systemctl --no-pager --lines=0 status "$SERVICE" || true
echo

# 5. Nudge if the admin password hasn't been set yet.
if ! "$PREFIX/.venv/bin/python" - <<'PY' 2>/dev/null
import sys
from sparkdash import store, auth
store.init_db()
sys.exit(0 if auth.admin_configured() else 1)
PY
then
  echo "NOTE: no admin password set yet. Set one with:"
  echo "  $PREFIX/.venv/bin/python -m sparkdash.admin set-password"
fi

echo "Done. https://$(hostname -f):${SPARKDASH_PORT:-7862}"
echo "Logs: journalctl -u $SERVICE -f"
