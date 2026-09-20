"""
add remaining missing columns (audit_logs.request_id, permissions.revoked_at)

Revision ID: 0003_more_missing_columns
Revises: 0002_user_security_columns
Create Date: 2026-07-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_more_missing_columns"
down_revision: str | None = "0002_user_security_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return column in {col["name"] for col in inspector.get_columns(table)}


def _index_exists(table: str, index: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return index in {ix["name"] for ix in inspector.get_indexes(table)}


def upgrade() -> None:
    # Found via a systematic diff of every SQLAlchemy model's columns
    # against 0001_initial_schema.py after the token_version bug (0002)
    # turned out not to be the only instance of this pattern. Both of
    # these were confirmed as real runtime failures, not just discovered
    # by inspection: audit_logs.request_id crashed the very first
    # register() call end-to-end (audit logging is on the write path for
    # nearly every mutating endpoint), and permissions.revoked_at is what
    # PermissionService.revoke() writes to.
    #
    # Existence checks (rather than a bare op.add_column) make this safe
    # to run even if alembic_version is out of sync with the actual
    # schema — e.g. an interrupted previous run, or a dev SQLite file
    # that survived across code updates.
    if not _column_exists("audit_logs", "request_id"):
        op.add_column(
            "audit_logs",
            sa.Column("request_id", sa.String(length=64), nullable=True),
        )
    if not _index_exists("audit_logs", "ix_audit_logs_request_id"):
        op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])

    if not _column_exists("permissions", "revoked_at"):
        op.add_column(
            "permissions",
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        )
    if not _index_exists("permissions", "ix_permissions_revoked_at"):
        op.create_index("ix_permissions_revoked_at", "permissions", ["revoked_at"])


def downgrade() -> None:
    op.drop_index("ix_permissions_revoked_at", table_name="permissions")
    op.drop_column("permissions", "revoked_at")

    op.drop_index("ix_audit_logs_request_id", table_name="audit_logs")
    op.drop_column("audit_logs", "request_id")
