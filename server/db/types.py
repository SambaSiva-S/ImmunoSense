"""Dialect-aware column types and the PHI-encryption seam.

Two concerns:

1. PORTABLE TYPES — production runs on Postgres (JSONB, native UUID), but tests
   run on SQLite. SQLAlchemy's TypeDecorator + with_variant lets one model
   definition use the right native type per engine. We define JSONType and
   GUID so the models read cleanly and work on both.

2. THE PHI ENCRYPTION SEAM (HIPAA-ready, not HIPAA-required in Phase 1).
   EncryptedString is a column type that, in Phase 1, stores plaintext (a
   no-op passthrough). The seam exists so that flipping on real encryption
   later is a configuration change — set IMMUNOSENSE_PHI_KEY and the same
   columns transparently encrypt/decrypt, with NO schema change and NO model
   change. This is the concrete realization of "design for HIPAA now, enable
   later" without paying the cost up front.
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy import CHAR, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.types import JSON


class GUID(TypeDecorator):
    """Platform-independent UUID: native UUID on Postgres, CHAR(36) on others."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# JSONB on Postgres, generic JSON on SQLite/others.
JSONType = JSON().with_variant(JSONB(), "postgresql")


class EncryptedString(TypeDecorator):
    """PHI column type with a transparent encryption seam.

    Phase 1: passthrough (stores plaintext). The column is MARKED as PHI by
    using this type, so a future audit can enumerate every PHI column, and
    enabling encryption is a single switch — no migration, no model edit.

    To enable encryption later: set IMMUNOSENSE_PHI_KEY and implement _encrypt
    / _decrypt (e.g. Fernet). The bind/result hooks already route through them.
    """

    impl = String
    cache_ok = True

    # Marker attribute so tooling can find PHI columns: `col.type.is_phi`.
    is_phi = True

    def _key(self):
        return os.environ.get("IMMUNOSENSE_PHI_KEY")

    def _encrypt(self, plaintext: str) -> str:
        key = self._key()
        if not key:
            return plaintext  # Phase 1 no-op
        # Phase 2: real symmetric encryption goes here (e.g. Fernet(key)).
        # Intentionally not implemented in Phase 1 to avoid a false sense of
        # security; the seam is what matters now.
        raise NotImplementedError(
            "IMMUNOSENSE_PHI_KEY is set but encryption is not implemented yet. "
            "Implement EncryptedString._encrypt/_decrypt before enabling."
        )

    def _decrypt(self, ciphertext: str) -> str:
        key = self._key()
        if not key:
            return ciphertext  # Phase 1 no-op
        raise NotImplementedError(
            "IMMUNOSENSE_PHI_KEY is set but decryption is not implemented yet."
        )

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return self._encrypt(str(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return self._decrypt(value)
