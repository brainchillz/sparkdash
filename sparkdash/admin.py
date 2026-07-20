"""Admin CLI: set the dashboard password out-of-band.

    python -m sparkdash.admin set-password [--username sparkadmin]

Reads the password from the SPARKDASH_ADMIN_PASSWORD env var if set (handy for
first boot / automation), otherwise prompts twice without echo.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from . import auth, config


def _set_password(username: str) -> int:
    pw = os.environ.get("SPARKDASH_ADMIN_PASSWORD")
    if pw:
        print(f"Using password from SPARKDASH_ADMIN_PASSWORD for '{username}'.")
    else:
        pw = getpass.getpass(f"New password for '{username}': ")
        if pw != getpass.getpass("Confirm password: "):
            print("Passwords did not match.", file=sys.stderr)
            return 1
    if len(pw) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        return 1
    auth.set_password(username, pw)
    print(f"Password set for '{username}'. Stored (scrypt) at {config.DB_FILE}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparkdash.admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("set-password", help="Set/reset the admin password.")
    sp.add_argument("--username", default=config.ADMIN_USER)
    args = parser.parse_args(argv)
    if args.cmd == "set-password":
        return _set_password(args.username)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
