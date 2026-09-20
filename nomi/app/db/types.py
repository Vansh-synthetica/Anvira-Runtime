from enum import StrEnum
from typing import TypeVar

from sqlalchemy import Enum

E = TypeVar("E", bound=StrEnum)


def str_enum_column(enum_class: type[E], **kwargs: object) -> Enum:
    """Map StrEnum values (lowercase) to PostgreSQL enum columns consistently."""
    return Enum(
        enum_class,
        values_callable=lambda members: [member.value for member in members],
        native_enum=True,
        **kwargs,
    )
