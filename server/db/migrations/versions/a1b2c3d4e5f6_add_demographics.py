"""add demographics (sex, height_cm, weight_kg) to profiles

Revision ID: a1b2c3d4e5f6
Revises: bb6b63ad1b79
Create Date: 2026-05-31

These feed the biomarker and dietary agents' percentile baselines. Stored
canonical (cm, kg); BMI is derived at evaluation time, never stored.
"""
from alembic import op
import sqlalchemy as sa


revision = "a1b2c3d4e5f6"
down_revision = "bb6b63ad1b79"
branch_labels = None
depends_on = None

SCHEMA = "identity"


def upgrade() -> None:
    bind = op.get_bind()
    # On SQLite (tests) there are no named schemas; pass schema=None there.
    schema = SCHEMA if bind.dialect.name != "sqlite" else None
    op.add_column("profiles", sa.Column("sex", sa.Integer(), nullable=True), schema=schema)
    op.add_column("profiles", sa.Column("height_cm", sa.Float(), nullable=True), schema=schema)
    op.add_column("profiles", sa.Column("weight_kg", sa.Float(), nullable=True), schema=schema)


def downgrade() -> None:
    bind = op.get_bind()
    schema = SCHEMA if bind.dialect.name != "sqlite" else None
    op.drop_column("profiles", "weight_kg", schema=schema)
    op.drop_column("profiles", "height_cm", schema=schema)
    op.drop_column("profiles", "sex", schema=schema)
