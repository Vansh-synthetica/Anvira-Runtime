"""Anvira Runtime version and capability constants.

Applications depend on ``API_VERSION`` and the capability list, never on the
version of ORCHA / Nomi / AICL that happens to sit behind the runtime.
"""
from __future__ import annotations

#: Semantic version of the runtime distribution (daemon + CLI + API).
RUNTIME_VERSION = "1.0.0"

#: Major version of the HTTP API. Bumped only on breaking API changes.
#: All routes live under ``/v{API_VERSION}``.
API_VERSION = 1

#: Minor revision of the API (additive changes only within one API_VERSION).
API_REVISION = 0

#: Capability flags an application can feature-detect via ``GET /version``.
CAPABILITIES = (
    "chat",
    "chat.stream",
    "chat.memory-recall",
    "models.catalog",
    "models.install",
    "models.select",
    "models.providers",
    "hardware",
    "orcha.run",
    "orcha.jobs",
    "orcha.cancel",
    "memory.store",
    "memory.search",
    "memory.namespaces",
    "apps.registry",
    "aicl.bus",
)

SERVICE_NAME = "anvira-runtime"


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH`` (extra suffixes such as ``-rc1`` ignored)."""
    core = text.strip().lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    nums: list[int] = []
    for part in parts[:3]:
        try:
            nums.append(int(part))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def satisfies(version: str, minimum: str) -> bool:
    """True when ``version >= minimum``."""
    return parse_version(version) >= parse_version(minimum)
