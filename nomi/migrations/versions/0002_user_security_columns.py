"""
add missing user security columns

Revision ID: 0002_user_security_columns
Revises: 0001_initial_schema
Create Date: 2026-07-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_user_security_columns"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_exists(table: str, column: str) -> bool:
    """
    Checks the live schema directly rather than trusting alembic_version
    to be perfectly in sync with it. This makes each add_column below safe
    to run even if a previous run was interrupted partway through, or a
    dev SQLite file survived across code updates with its schema ahead of
    what alembic_version recorded — both real scenarios encountered while
    testing the desktop launch path.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return column in {col["name"] for col in inspector.get_columns(table)}


def upgrade() -> None:
    # These columns exist on app.models.user.User but were missing from the
    # initial migration, causing every query against `users` to fail with
    # asyncpg.exceptions.UndefinedColumnError: column users.token_version
    # does not exist.
    if not _column_exists("users", "token_version"):
        op.add_column(
            "users",
            sa.Column("token_version", sa.Integer(), server_default="1", nullable=False),
        )
    if not _column_exists("users", "failed_login_count"):
        op.add_column(
            "users",
            sa.Column(
                "failed_login_count", sa.Integer(), server_default="0", nullable=False
            ),
        )
    if not _column_exists("users", "locked_until"):
        op.add_column(
            "users",
            sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_login_count")
    op.drop_column("users", "token_version")
