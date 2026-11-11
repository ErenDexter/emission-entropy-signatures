"""Cryptographic attestation of enrollment records ("the ledger").

The mint signs every enrolled fingerprint with an Ed25519 private key. Verifiers only
hold the public key: before trusting a stored record they recompute its digest and
check the signature, so an attacker who gains write access to the database cannot
insert a counterfeit's fingerprint or swap one note's pattern for another.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def record_digest(*fields: str | bytes) -> bytes:
    """SHA-256 over length-prefixed fields (unambiguous concatenation)."""
    h = hashlib.sha256()
    for f in fields:
        b = f.encode() if isinstance(f, str) else f
        h.update(struct.pack(">Q", len(b)))
        h.update(b)
    return h.digest()


class Ledger:
    PRIVATE = "mint_private.pem"
    PUBLIC = "mint_public.pem"

    def __init__(self, keys_dir: str | Path, create: bool = False):
        keys_dir = Path(keys_dir)
        priv_path, pub_path = keys_dir / self.PRIVATE, keys_dir / self.PUBLIC
        self._private: Ed25519PrivateKey | None = None
        if priv_path.exists():
            self._private = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
        elif create and not pub_path.exists():
            # Only a brand-new installation generates keys. A verifier that holds just the
            # mint's public key must never replace it with one of its own.
            keys_dir.mkdir(parents=True, exist_ok=True)
            self._private = Ed25519PrivateKey.generate()
            priv_path.write_bytes(self._private.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
            pub_path.write_bytes(self._private.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
        if pub_path.exists():
            self._public: Ed25519PublicKey = serialization.load_pem_public_key(pub_path.read_bytes())
        elif self._private is not None:
            self._public = self._private.public_key()
        else:
            raise FileNotFoundError(f"no mint public key in {keys_dir}")

    @property
    def can_sign(self) -> bool:
        return self._private is not None

    def sign(self, digest: bytes) -> bytes:
        if self._private is None:
            raise PermissionError("this installation holds only the public key; enrollment happens at the mint")
        return self._private.sign(digest)

    def verify(self, digest: bytes, signature: bytes) -> bool:
        try:
            self._public.verify(signature, digest)
            return True
        except InvalidSignature:
            return False

    def public_key_id(self) -> str:
        raw = self._public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return hashlib.sha256(raw).hexdigest()[:16]
