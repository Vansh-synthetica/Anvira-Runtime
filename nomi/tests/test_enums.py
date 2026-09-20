"""Regression tests for PostgreSQL enum alignment."""

from app.core.enums import (
    MemoryType,
    MemoryVisibility,
    PermissionLevel,
    PermissionTargetType,
    UserRole,
)
from app.db.types import str_enum_column


def test_str_enum_values_are_lowercase() -> None:
    assert UserRole.USER.value == "user"
    assert UserRole.ADMIN.value == "admin"
    assert MemoryType.NOTE.value == "note"
    assert MemoryVisibility.PRIVATE.value == "private"
    assert PermissionLevel.READ.value == "read"
    assert PermissionTargetType.MEMORY.value == "memory"


def test_str_enum_column_uses_values_not_names() -> None:
    column = str_enum_column(UserRole)
    assert column.enums == ["user", "admin"]
