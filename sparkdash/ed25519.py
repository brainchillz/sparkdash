"""Ed25519 signature verification — pure stdlib, no dependencies.

This file is deliberately dependency-free and is COPIED VERBATIM into every
app that verifies SSO assertions (nexusdash/core/ed25519.py,
nexusipam/core/ed25519.py, dnsmaqmgr/core/ed25519.py). That is the whole point
of it existing: `cryptography` is not a dependency of any of those apps, and
adding one would mean pushing a new dependency to every host that runs them —
a needless risk for something verification does not require.

Scope, deliberately narrow:
  * VERIFY ONLY. There is no signing here and there must never be. Signing
    touches secret key material, where a non-constant-time pure-Python
    implementation would be the wrong tool; the issuer signs with the
    `cryptography` package instead (it is a standalone service, so its
    dependencies cost nothing at the fleet level).
  * Verification handles only public values — a public key, a message and a
    signature an attacker already holds — so the timing-channel objection that
    rules out hand-rolled signing does not apply.

Algorithm is the RFC 8032 §6 reference implementation, points in extended
homogeneous coordinates. Correctness is pinned by the RFC 8032 §7.1 test
vectors in tests/test_ed25519.py — including the negative cases, which is what
actually catches a broken verifier (a verifier that returns True for
everything passes every positive vector).

Python 3.9 through 3.14 clean — the oldest and newest interpreters this has
to run on.
"""
import hashlib

# Curve25519 field prime and group order.
_P = 2 ** 255 - 19
_Q = 2 ** 252 + 27742317777372353535851937790883648493


def _modp_inv(x):
    return pow(x, _P - 2, _P)


_D = -121665 * _modp_inv(121666) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _sha512(b):
    return hashlib.sha512(b).digest()


def _sha512_modq(b):
    return int.from_bytes(_sha512(b), 'little') % _Q


def _recover_x(y, sign):
    """Recover the x coordinate of a compressed point, or None if the encoding
    does not name a curve point at all."""
    if y >= _P:
        return None
    x2 = (y * y - 1) * _modp_inv(_D * y * y + 1) % _P
    if x2 == 0:
        # y = ±1. Only the even root is a valid encoding here.
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None            # not a square: not on the curve
    if (x & 1) != sign:
        x = _P - x
    return x


# Base point, in extended coordinates (X, Y, Z, T) with x=X/Z, y=Y/Z, xy=T/Z.
_GY = 4 * _modp_inv(5) % _P
_GX = _recover_x(_GY, 0)
_G = (_GX, _GY, 1, _GX * _GY % _P)
_IDENTITY = (0, 1, 1, 0)


def _point_add(p, q):
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * p[3] * q[3] * _D % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _point_mul(s, p):
    """Double-and-add. Not constant time — see the module docstring: every
    input here is public."""
    out = _IDENTITY
    while s > 0:
        if s & 1:
            out = _point_add(out, p)
        p = _point_add(p, p)
        s >>= 1
    return out


def _point_equal(p, q):
    # Projective coordinates: compare cross-multiplied affine values.
    if (p[0] * q[2] - q[0] * p[2]) % _P != 0:
        return False
    if (p[1] * q[2] - q[1] * p[2]) % _P != 0:
        return False
    return True


def _point_decompress(s):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, 'little')
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def verify(public_key, message, signature):
    """True if `signature` is a valid Ed25519 signature over `message` by
    `public_key`. Never raises on malformed input — a bad key, a bad length or
    a non-curve point is simply False, so callers can hand it untrusted bytes.

    All three arguments are bytes.
    """
    if not isinstance(public_key, (bytes, bytearray)) or len(public_key) != 32:
        return False
    if not isinstance(signature, (bytes, bytearray)) or len(signature) != 64:
        return False
    a = _point_decompress(bytes(public_key))
    if a is None:
        return False
    rs = bytes(signature[:32])
    r = _point_decompress(rs)
    if r is None:
        return False
    s = int.from_bytes(signature[32:], 'little')
    if s >= _Q:
        # Non-canonical scalar. Rejecting this is what stops trivial signature
        # malleability (s and s+L would otherwise both verify).
        return False
    h = _sha512_modq(rs + bytes(public_key) + bytes(message))
    return _point_equal(_point_mul(s, _G), _point_add(r, _point_mul(h, a)))
