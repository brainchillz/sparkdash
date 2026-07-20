"""TLS certificate management.

The app serves HTTPS from a cert/key pair on disk. On startup `ensure()`
guarantees a usable pair exists — validating the configured one and, if it is
missing/broken/expired, regenerating a self-signed fallback (after backing up
the bad files). That anti-lockout behaviour is what makes a self-restart safe
when an admin swaps in a custom cert: a bad cert can never brick access.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import ssl
import sys
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtensionOID, NameOID

from . import config


# -- generation --------------------------------------------------------------

def generate_self_signed() -> tuple[bytes, bytes]:
    """Return (cert_pem, key_pem) for a fresh self-signed cert."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "SparkDash")])

    sans: list[x509.GeneralName] = [x509.DNSName(h) for h in config.CERT_HOSTNAMES]
    for ip in config.CERT_IPS:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    return cert_pem, key_pem


# -- validation --------------------------------------------------------------

def validate(cert_pem: bytes, key_pem: bytes) -> tuple[bool, str]:
    """Check a cert/key pair is loadable, matched, and currently valid."""
    try:
        cert = x509.load_pem_x509_certificate(cert_pem)
    except Exception as exc:
        return False, f"certificate did not parse: {exc}"
    try:
        key = serialization.load_pem_private_key(key_pem, password=None)
    except Exception as exc:
        return False, f"private key did not parse: {exc}"

    cpub = cert.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    kpub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    if cpub != kpub:
        return False, "private key does not match certificate"

    now = dt.datetime.now(dt.timezone.utc)
    if cert.not_valid_after_utc < now:
        return False, "certificate has expired"
    if cert.not_valid_before_utc > now:
        return False, "certificate is not yet valid"

    # Final authority: does OpenSSL accept the pair together?
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False) as cf, \
                tempfile.NamedTemporaryFile(delete=False) as kf:
            cf.write(cert_pem); kf.write(key_pem)
            cpath, kpath = cf.name, kf.name
        try:
            ctx.load_cert_chain(cpath, kpath)
        finally:
            os.unlink(cpath); os.unlink(kpath)
    except Exception as exc:
        return False, f"OpenSSL rejected the pair: {exc}"

    return True, "ok"


# -- info --------------------------------------------------------------------

def info() -> dict:
    """Describe the currently installed cert for the admin UI."""
    try:
        cert_pem = config.CERT_FILE.read_bytes()
        cert = x509.load_pem_x509_certificate(cert_pem)
    except Exception:
        return {"present": False}

    def _cn(name: x509.Name) -> str:
        attrs = name.get_attributes_for_oid(NameOID.COMMON_NAME)
        return attrs[0].value if attrs else name.rfc4514_string()

    sans: list[str] = []
    try:
        ext = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME).value
        sans = [str(g.value) for g in ext]
    except x509.ExtensionNotFound:
        pass

    fp = cert.fingerprint(hashes.SHA256()).hex()
    fp = ":".join(fp[i:i+2] for i in range(0, len(fp), 2))
    self_signed = cert.issuer == cert.subject
    return {
        "present": True,
        "subject": _cn(cert.subject),
        "issuer": _cn(cert.issuer),
        "self_signed": self_signed,
        "sans": sans,
        "not_before": cert.not_valid_before_utc.isoformat(),
        "not_after": cert.not_valid_after_utc.isoformat(),
        "fingerprint_sha256": fp,
    }


# -- disk ops ----------------------------------------------------------------

def _write_pair(cert_pem: bytes, key_pem: bytes) -> None:
    config.CERT_DIR.mkdir(parents=True, exist_ok=True)
    config.CERT_FILE.write_bytes(cert_pem)
    config.CERT_FILE.chmod(0o644)
    config.KEY_FILE.write_bytes(key_pem)
    config.KEY_FILE.chmod(0o600)


def _backup_existing(reason: str) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in (config.CERT_FILE, config.KEY_FILE):
        if f.exists():
            f.rename(f.with_suffix(f.suffix + f".bad-{reason}-{stamp}"))


def install_custom(cert_pem: bytes, key_pem: bytes) -> None:
    """Validate then persist an admin-supplied cert/key. Raises on invalid."""
    ok, msg = validate(cert_pem, key_pem)
    if not ok:
        raise ValueError(msg)
    _write_pair(cert_pem, key_pem)


def ensure() -> None:
    """Guarantee a usable cert/key pair exists; self-heal if not."""
    if config.CERT_FILE.exists() and config.KEY_FILE.exists():
        ok, msg = validate(config.CERT_FILE.read_bytes(),
                           config.KEY_FILE.read_bytes())
        if ok:
            return
        print(f"[sparkdash] configured cert invalid ({msg}); "
              f"regenerating self-signed.", file=sys.stderr)
        _backup_existing("invalid")
    cert_pem, key_pem = generate_self_signed()
    _write_pair(cert_pem, key_pem)
    print(f"[sparkdash] wrote self-signed cert to {config.CERT_FILE}",
          file=sys.stderr)


if __name__ == "__main__":
    ensure()
