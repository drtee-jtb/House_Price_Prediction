"""add prediction reuse tracking

Revision ID: 20260413_000003
Revises: 20260413_000002
Create Date: 2026-04-13 00:00:03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260413_000003"
down_revision = "20260413_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.add_column(sa.Column("source_prediction_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("was_reused", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.create_index(op.f("ix_predictions_source_prediction_id"), ["source_prediction_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_index(op.f("ix_predictions_source_prediction_id"))
        batch_op.drop_column("was_reused")
        batch_op.drop_column("source_prediction_id")