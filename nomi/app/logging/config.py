import json
import logging
from typing import Any

SECRET_KEYS = {"authorization", "cookie", "password", "token", "secret", "api_key"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if key.lower() in SECRET_KEYS else redact(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id") and record.request_id:
            payload["request_id"] = record.request_id
        return json.dumps(redact(payload), default=str)


def configure_logging(level: str, environment: str = "development") -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    if environment == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

    root.addHandler(handler)
    root.setLevel(level.upper())