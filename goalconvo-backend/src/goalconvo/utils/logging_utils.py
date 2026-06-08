"""Centralized logging setup for batch runs and servers."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config import Config


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging_from_config(config: Optional["Config"] = None) -> None:
    level = logging.INFO
    if config is not None and getattr(config, "log_level", None):
        level = getattr(logging, str(config.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
