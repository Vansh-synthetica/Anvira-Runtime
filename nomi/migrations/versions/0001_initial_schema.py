"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_USER_ROLE = sa.Enum("user", "admin", name="userrole")
_MEMORY_TYPE = sa.Enum(
    "preference",
    "fact",
    "project",
    "person",
    "style",
    "instruction",
    "note",
    "research",
    "temporary",
    "session",
    "custom",
    name="memorytype",
)
_MEMORY_VISIBILITY = sa.Enum("private", "permissioned", "public", name="memoryvisibility")
_PERMISSION_TARGET = sa.Enum(
    "identity_profile",
    "memory_category",
    "memory",
    "connector",
    "workspace",
    name="permissiontargettype",
)
_PERMISSION_LEVEL = sa.Enum("read", "write", "admin", "deny", name="permissionlevel")


def upgrade() -> None:
    _USER_ROLE.create(op.get_bind(), checkfirst=True)
    _MEMORY_TYPE.create(op.get_bind(), checkfirst=True)
    _MEMORY_VISIBILITY.create(op.get_bind(), checkfirst=True)
    _PERMISSION_TARGET.create(op.get_bind(), checkfirst=True)
    _PERMISSION_LEVEL.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", _USER_ROLE, server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("is_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("token_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("failed_login_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "identity_profiles",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("display_name", sa.String(length=120)),
        sa.Column("pronouns", sa.String(length=64)),
        sa.Column("timezone", sa.String(length=64)),
        sa.Column("locale", sa.String(length=32)),
        sa.Column("preferred_language", sa.String(length=64)),
        sa.Column("writing_style", sa.Text()),
        sa.Column("tone", sa.String(length=120)),
        sa.Column("preferred_output_format", sa.String(length=120)),
        sa.Column("occupation", sa.String(length=160)),
        sa.Column("bio", sa.Text()),
        sa.Column("interests", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("goals", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("preferences", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("constraints", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("avatar_url", sa.String(length=2048)),
        sa.Column("extra", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_identity_profiles_user_id", "identity_profiles", ["user_id"], unique=True)

    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=240), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("memory_type", _MEMORY_TYPE, nullable=False),
        sa.Column("category", sa.String(length=120)),
        sa.Column("importance", sa.Integer(), server_default="3", nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column(
            "visibility",
            _MEMORY_VISIBILITY,
            server_default="private",
            nullable=False,
        ),
        sa.Column("source", sa.String(length=240)),
        sa.Column("tags", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("extra", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("is_archived", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("ix_memories_title", "memories", ["title"])
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"])
    op.create_index("ix_memories_category", "memories", ["category"])
    op.create_index("ix_memories_is_archived", "memories", ["is_archived"])
    op.create_index("ix_memories_deleted_at", "memories", ["deleted_at"])

    op.create_table(
        "permissions",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("application_name", sa.String(length=160), nullable=False),
        sa.Column("application_id", sa.String(length=160)),
        sa.Column("target_type", _PERMISSION_TARGET, nullable=False),
        sa.Column("target_id", sa.String(length=160)),
        sa.Column("permission_level", _PERMISSION_LEVEL, nullable=False),
        sa.Column("scope", sa.String(length=240)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_permissions_user_id", "permissions", ["user_id"])
    op.create_index("ix_permissions_application_name", "permissions", ["application_name"])
    op.create_index("ix_permissions_application_id", "permissions", ["application_id"])
    op.create_index("ix_permissions_target_type", "permissions", ["target_type"])
    op.create_index("ix_permissions_target_id", "permissions", ["target_id"])
    op.create_index("ix_permissions_scope", "permissions", ["scope"])
    op.create_index("ix_permissions_is_active", "permissions", ["is_active"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("actor_user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action_type", sa.String(length=120), nullable=False),
        sa.Column("resource_type", sa.String(length=120)),
        sa.Column("resource_id", sa.String(length=160)),
        sa.Column("extra", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("ip_address", sa.String(length=64)),
        sa.Column("user_agent", sa.String(length=512)),
        sa.Column("request_id", sa.String(length=64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_action_type", "audit_logs", ["action_type"])
    op.create_index("ix_audit_logs_resource_type", "audit_logs", ["resource_type"])
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"])
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("permissions")
    op.drop_table("memories")
    op.drop_table("identity_profiles")
    op.drop_table("users")
    _PERMISSION_LEVEL.drop(op.get_bind(), checkfirst=True)
    _PERMISSION_TARGET.drop(op.get_bind(), checkfirst=True)
    _MEMORY_VISIBILITY.drop(op.get_bind(), checkfirst=True)
    _MEMORY_TYPE.drop(op.get_bind(), checkfirst=True)
    _USER_ROLE.drop(op.get_bind(), checkfirst=True)
