"""add feature policy metadata to prediction requests

Revision ID: 20260415_000006
Revises: 20260415_000005
Create Date: 2026-04-15 00:00:06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260415_000006"
down_revision = "20260415_000005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("prediction_requests") as batch_op:
        batch_op.add_column(
            sa.Column(
                "feature_policy_name",
                sa.String(length=80),
                nullable=False,
                server_default="balanced-v1",
            )
        )
        batch_op.add_column(
            sa.Column(
                "feature_policy_version",
                sa.String(length=30),
                nullable=False,
                server_default="v1",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("prediction_requests") as batch_op:
        batch_op.drop_column("feature_policy_version")
        batch_op.drop_column("feature_policy_name")
