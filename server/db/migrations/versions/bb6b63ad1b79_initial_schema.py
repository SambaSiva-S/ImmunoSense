"""initial schema (metadata-driven, schema-aware)

Revision ID: bb6b63ad1b79
Revises:
Create Date: 2026-05-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import server.db.types
from sqlalchemy import Text

revision: str = "bb6b63ad1b79"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SCHEMAS = ("identity", "health", "reference", "audit")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for schema in _SCHEMAS:
            op.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    from server.db.base import Base
    from server.db import models  # noqa: F401
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    from server.db.base import Base
    from server.db import models  # noqa: F401
    Base.metadata.drop_all(bind=bind)
    if bind.dialect.name == "postgresql":
        for schema in _SCHEMAS:
            op.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')