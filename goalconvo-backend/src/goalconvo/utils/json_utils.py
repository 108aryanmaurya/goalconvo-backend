"""JSON and filesystem helpers for datasets and experiment artifacts."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


class _DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def load_json(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("File not found: %s", file_path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", file_path, e)
        return {}


def save_json(data: Dict[str, Any], file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_DateTimeEncoder)
