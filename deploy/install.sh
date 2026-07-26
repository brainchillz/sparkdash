#!/usr/bin/env bash
# Install (or update) SparkDash into a dedicated directory and run it as a
# systemd service. Deploys from this checkout — safe to re-run to update.
#
#   ./deploy/install.sh
#   ./deploy/install.sh --set-password          # also set the admin password
#   SPARKDASH_PREFIX=/srv/sparkdash ./deploy/install.sh
#
# --set-password prompts twice without echo, or reads SPARKDASH_ADMIN_PASSWORD
# from the environment for unattended installs.
#
# Filesystem layout (all outside any user's home directory):
#   /opt/sparkdash        runtime files only: package, frontend, lockfiles, venv
#   /etc/sparkdash        config.toml (+ refreshed config.toml.example)
#   /var/lib/sparkdash    state: password DB, TLS certs, metrics history —
#                         never touched by install/update, so these persist.
# Pre-existing state/config under the installing user's home is migrated in
# on first install (the home copy is left behind as a backup).
set -euo pipefail

SET_PASSWORD=0
for arg in "$@"; do
  case "$arg" in
    --set-password) SET_PASSWORD=1 ;;
    -h|--help)
      sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Running the whole script as root is the classic footgun: everything ends up
# root-owned and unreadable/unwritable for the service user.
if [[ "$(id -u)" == 0 ]]; then
  echo "error: run as the service user (e.g. nvidia), not root — sudo is used internally where needed." >&2
  exit 1
fi

PREFIX="${SPARKDASH_PREFIX:-/opt/sparkdash}"
ETC_DIR="/etc/sparkdash"
STATE_DIR="/var/lib/sparkdash"
SERVICE="sparkdash.service"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
SVC_USER="$(id -un)"          # the service runs as whoever installs it
SVC_GROUP="$(id -gn)"

UV="$(command -v uv || true)"
[[ -n "$UV" ]] || { echo "error: 'uv' not found in PATH." >&2; exit 1; }

echo "SparkDash install"
echo "  source : $SRC"
echo "  target : $PREFIX"
echo "  config : $ETC_DIR    state: $STATE_DIR"
echo "  service: $SERVICE (user: $SVC_USER)"
echo

# Reminder: install on the Ray head node — Ray's dashboard binds to 127.0.0.1
# there, so Ray/vLLM data is only available on the head.
echo "note: install on the Ray head node (Ray's dashboard is bound to localhost)."
echo

# 1. Install dir, owned by the service user so the rest needs no sudo.
sudo mkdir -p "$PREFIX"
sudo chown "$SVC_USER":"$SVC_GROUP" "$PREFIX"

# 2. Mirror in ONLY what the app needs at runtime (package, frontend, project
#    files for the venv build). Docs, deploy tooling, examples and secrets
#    stay out; anything else already in $PREFIX is removed. The venv is
#    protected so updates don't rebuild it from scratch.
rsync -a --delete --delete-excluded \
  --filter='protect /.venv' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --include '/sparkdash/***' --include '/frontend/***' \
  --include '/pyproject.toml' --include '/uv.lock' \
  --exclude '*' \
  "$SRC/" "$PREFIX/"

# 3. Build an isolated venv in the install dir from the lockfile.
( cd "$PREFIX" && "$UV" sync --frozen )

# 4. Config dir: refresh the example; migrate a home-dir config on first
#    install; never overwrite an existing /etc config.
sudo mkdir -p "$ETC_DIR"
sudo install -m 0644 "$SRC/sparkdash.example.toml" "$ETC_DIR/config.toml.example"
if [[ ! -f "$ETC_DIR/config.toml" && -f "$HOME/.config/sparkdash/config.toml" ]]; then
  echo "Migrating config: ~/.config/sparkdash/config.toml -> $ETC_DIR/config.toml"
  sudo install -m 0644 "$HOME/.config/sparkdash/config.toml" "$ETC_DIR/config.toml"
fi

# 5. State dir, owned by the service user. Migrate pre-existing home-dir state
#    (password DB, certs, history) the first time; the home copy stays behind
#    as a backup.
sudo mkdir -p "$STATE_DIR"
sudo chown "$SVC_USER":"$SVC_GROUP" "$STATE_DIR"
OLD_STATE="$HOME/.local/share/sparkdash"
if [[ ! -f "$STATE_DIR/sparkdash.db" && -f "$OLD_STATE/sparkdash.db" ]]; then
  echo "Migrating state: $OLD_STATE -> $STATE_DIR (home copy kept as backup)"
  cp -a "$OLD_STATE/." "$STATE_DIR/"
fi

# 6. Install the unit from the source checkout, substituting PREFIX, then
#    enable + (re)start it.
sudo sed "s#/opt/sparkdash#$PREFIX#g" "$SRC/deploy/$SERVICE" \
  | sudo tee "/etc/systemd/system/$SERVICE" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"     # restart so updates take effect

echo
sudo systemctl --no-pager --lines=0 status "$SERVICE" || true
echo

# 7. Optionally set the admin password now. A failed prompt (mismatch, too
#    short, Ctrl-C) must not abort an otherwise-finished install.
# (cd first: the package is resolved from the working directory, not the venv)
if [[ "$SET_PASSWORD" == 1 ]]; then
  echo "Setting admin password…"
  ( cd "$PREFIX" && .venv/bin/python -m sparkdash.admin set-password ) \
    || echo "warning: password not set — run: cd $PREFIX && .venv/bin/python -m sparkdash.admin set-password"
fi

# 8. Nudge if the admin password hasn't been set yet.
if ! ( cd "$PREFIX" && .venv/bin/python - <<'PY' 2>/dev/null
import sys
from sparkdash import store, auth
store.init_db()
sys.exit(0 if auth.admin_configured() else 1)
PY
)
then
  echo "NOTE: no admin password set yet. Set one with:"
  echo "  cd $PREFIX && .venv/bin/python -m sparkdash.admin set-password"
fi

echo "Done. https://$(hostname -f):${SPARKDASH_PORT:-7862}"
echo "Logs: journalctl -u $SERVICE -f"
