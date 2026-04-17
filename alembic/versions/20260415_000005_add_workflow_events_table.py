"""add workflow events table

Revision ID: 20260415_000005
Revises: 20260413_000004
Create Date: 2026-04-15 00:00:05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260415_000005"
down_revision = "20260413_000004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "workflow_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("event_name", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["prediction_requests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_workflow_events_request_id"), "workflow_events", ["request_id"], unique=False)
    op.create_index(op.f("ix_workflow_events_event_name"), "workflow_events", ["event_name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_workflow_events_event_name"), table_name="workflow_events")
    op.drop_index(op.f("ix_workflow_events_request_id"), table_name="workflow_events")
    op.drop_table("workflow_events")
