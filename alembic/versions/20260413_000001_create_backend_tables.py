"""create backend tables

Revision ID: 20260413_000001
Revises:
Create Date: 2026-04-13 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260413_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_requests",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("correlation_id", sa.String(length=36), nullable=False),
        sa.Column("requested_by", sa.String(length=120), nullable=True),
        sa.Column("address_line_1", sa.String(length=120), nullable=False),
        sa.Column("address_line_2", sa.String(length=120), nullable=True),
        sa.Column("city", sa.String(length=80), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("postal_code", sa.String(length=20), nullable=False),
        sa.Column("country", sa.String(length=60), nullable=False),
        sa.Column("normalized_address", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_prediction_requests_correlation_id"),
        "prediction_requests",
        ["correlation_id"],
        unique=False,
    )

    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.String(length=60), nullable=False),
        sa.Column("completeness_score", sa.Float(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["prediction_requests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_feature_snapshots_request_id"),
        "feature_snapshots",
        ["request_id"],
        unique=False,
    )

    op.create_table(
        "provider_responses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("provider_name", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["prediction_requests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_provider_responses_request_id"),
        "provider_responses",
        ["request_id"],
        unique=False,
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("feature_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("predicted_price", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=10), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.String(length=60), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["feature_snapshot_id"], ["feature_snapshots.id"]),
        sa.ForeignKeyConstraint(["request_id"], ["prediction_requests.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_predictions_request_id"), "predictions", ["request_id"], unique=False)
    op.create_index(
        op.f("ix_predictions_feature_snapshot_id"),
        "predictions",
        ["feature_snapshot_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_feature_snapshot_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_request_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_provider_responses_request_id"), table_name="provider_responses")
    op.drop_table("provider_responses")
    op.drop_index(op.f("ix_feature_snapshots_request_id"), table_name="feature_snapshots")
    op.drop_table("feature_snapshots")
    op.drop_index(op.f("ix_prediction_requests_correlation_id"), table_name="prediction_requests")
    op.drop_table("prediction_requests")