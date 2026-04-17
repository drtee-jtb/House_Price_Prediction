"""add index on predictions.generated_at

Revision ID: 20260416_000007
Revises: 20260415_000006
Create Date: 2026-04-16 00:00:07
"""

from __future__ import annotations

from alembic import op


revision = "20260416_000007"
down_revision = "20260415_000006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.create_index("ix_predictions_generated_at", ["generated_at"])


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_index("ix_predictions_generated_at")
