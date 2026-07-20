"""Entrypoint: ensure a TLS cert exists, then serve HTTPS.

    python -m sparkdash

Used by both run.sh and the systemd unit. A clean exit (e.g. after an in-app
cert swap sends SIGTERM) lets systemd restart us with the new cert.
"""

from __future__ import annotations

import uvicorn

from . import certs, config, store


def main() -> None:
    store.init_db()
    certs.ensure()
    uvicorn.run(
        "sparkdash.app:app",
        host="0.0.0.0",
        port=config.PORT,
        ssl_certfile=str(config.CERT_FILE),
        ssl_keyfile=str(config.KEY_FILE),
    )


if __name__ == "__main__":
    main()
