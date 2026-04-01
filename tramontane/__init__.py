"""Tramontane — Mistral-native agent orchestration framework."""

from __future__ import annotations

import logging

__version__ = "0.1.5"
__author__ = "Bleucommerce SAS"
__license__ = "MIT"

# Library best practice: let the user configure logging.
logging.getLogger("tramontane").addHandler(logging.NullHandler())

# Public API — convenience imports (after logging setup intentionally)
from tramontane.core.agent import Agent, AgentResult, StreamEvent  # noqa: E402
from tramontane.core.pipeline import Pipeline  # noqa: E402
from tramontane.router.router import MistralRouter  # noqa: E402

__all__ = [
    "Agent",
    "AgentResult",
    "StreamEvent",
    "Pipeline",
    "MistralRouter",
    "__version__",
]
