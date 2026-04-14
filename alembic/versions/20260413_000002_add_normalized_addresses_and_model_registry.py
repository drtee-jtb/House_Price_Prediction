"""add normalized addresses and model registry

Revision ID: 20260413_000002
Revises: 20260413_000001
Create Date: 2026-04-13 00:00:02
"""

from __future__ import annotations

from uuid import uuid4

from alembic import op
import sqlalchemy as sa


revision = "20260413_000002"
down_revision = "20260413_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "normalized_addresses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("address_line_1", sa.String(length=120), nullable=False),
        sa.Column("address_line_2", sa.String(length=120), nullable=True),
        sa.Column("city", sa.String(length=80), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("postal_code", sa.String(length=20), nullable=False),
        sa.Column("country", sa.String(length=60), nullable=False),
        sa.Column("formatted_address", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("formatted_address"),
    )
    op.create_index(
        op.f("ix_normalized_addresses_formatted_address"),
        "normalized_addresses",
        ["formatted_address"],
        unique=True,
    )

    op.create_table(
        "model_registry",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=120), nullable=False),
        sa.Column("model_version", sa.String(length=60), nullable=False),
        sa.Column("feature_columns", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_registry_model_name"), "model_registry", ["model_name"], unique=False)
    op.create_index(
        op.f("ix_model_registry_model_version"),
        "model_registry",
        ["model_version"],
        unique=False,
    )

    op.add_column(
        "prediction_requests",
        sa.Column("normalized_address_id", sa.String(length=36), nullable=True),
    )

    connection = op.get_bind()
    request_rows = connection.execute(
        sa.text(
            "SELECT id, address_line_1, address_line_2, city, state, postal_code, country, normalized_address FROM prediction_requests"
        )
    ).mappings()
    for row in request_rows:
        normalized_address_id = str(uuid4())
        connection.execute(
            sa.text(
                "INSERT INTO normalized_addresses (id, address_line_1, address_line_2, city, state, postal_code, country, formatted_address, created_at, updated_at) "
                "VALUES (:id, :address_line_1, :address_line_2, :city, :state, :postal_code, :country, :formatted_address, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {
                "id": normalized_address_id,
                "address_line_1": row["address_line_1"],
                "address_line_2": row["address_line_2"],
                "city": row["city"],
                "state": row["state"],
                "postal_code": row["postal_code"],
                "country": row["country"],
                "formatted_address": row["normalized_address"],
            },
        )
        connection.execute(
            sa.text(
                "UPDATE prediction_requests SET normalized_address_id = :normalized_address_id WHERE id = :request_id"
            ),
            {"normalized_address_id": normalized_address_id, "request_id": row["id"]},
        )

    with op.batch_alter_table("prediction_requests") as batch_op:
        batch_op.alter_column("normalized_address_id", nullable=False)
        batch_op.create_index(
            op.f("ix_prediction_requests_normalized_address_id"),
            ["normalized_address_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_prediction_requests_normalized_address_id",
            "normalized_addresses",
            ["normalized_address_id"],
            ["id"],
        )

    op.add_column(
        "predictions",
        sa.Column("model_registry_id", sa.String(length=36), nullable=True),
    )
    default_registry_id = str(uuid4())
    connection.execute(
        sa.text(
            "INSERT INTO model_registry (id, model_name, model_version, feature_columns, is_active, created_at, updated_at) "
            "VALUES (:id, :model_name, :model_version, :feature_columns, :is_active, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "id": default_registry_id,
            "model_name": "legacy-model",
            "model_version": "legacy",
            "feature_columns": "[]",
            "is_active": True,
        },
    )
    connection.execute(
        sa.text("UPDATE predictions SET model_registry_id = :model_registry_id"),
        {"model_registry_id": default_registry_id},
    )
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.alter_column("model_registry_id", nullable=False)
        batch_op.create_index(
            op.f("ix_predictions_model_registry_id"),
            ["model_registry_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_predictions_model_registry_id",
            "model_registry",
            ["model_registry_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.drop_constraint("fk_predictions_model_registry_id", type_="foreignkey")
        batch_op.drop_index(op.f("ix_predictions_model_registry_id"))
        batch_op.drop_column("model_registry_id")
    with op.batch_alter_table("prediction_requests") as batch_op:
        batch_op.drop_constraint("fk_prediction_requests_normalized_address_id", type_="foreignkey")
        batch_op.drop_index(op.f("ix_prediction_requests_normalized_address_id"))
        batch_op.drop_column("normalized_address_id")
    op.drop_index(op.f("ix_model_registry_model_version"), table_name="model_registry")
    op.drop_index(op.f("ix_model_registry_model_name"), table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_index(
        op.f("ix_normalized_addresses_formatted_address"),
        table_name="normalized_addresses",
    )
    op.drop_table("normalized_addresses")