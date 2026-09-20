from enum import StrEnum


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    FACT = "fact"
    PROJECT = "project"
    PERSON = "person"
    STYLE = "style"
    INSTRUCTION = "instruction"
    NOTE = "note"
    RESEARCH = "research"
    TEMPORARY = "temporary"
    SESSION = "session"
    CUSTOM = "custom"


class MemoryVisibility(StrEnum):
    PRIVATE = "private"
    PERMISSIONED = "permissioned"
    PUBLIC = "public"


class PermissionLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    DENY = "deny"


class PermissionTargetType(StrEnum):
    IDENTITY_PROFILE = "identity_profile"
    MEMORY_CATEGORY = "memory_category"
    MEMORY = "memory"
    CONNECTOR = "connector"
    WORKSPACE = "workspace"

