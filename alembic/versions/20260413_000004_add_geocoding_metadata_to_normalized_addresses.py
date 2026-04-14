"""add geocoding metadata to normalized addresses

Revision ID: 20260413_000004
Revises: 20260413_000003
Create Date: 2026-04-13 00:00:04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260413_000004"
down_revision = "20260413_000003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("normalized_addresses") as batch_op:
        batch_op.add_column(sa.Column("latitude", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("longitude", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("geocoding_source", sa.String(length=60), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("normalized_addresses") as batch_op:
        batch_op.drop_column("geocoding_source")
        batch_op.drop_column("longitude")
        batch_op.drop_column("latitude")