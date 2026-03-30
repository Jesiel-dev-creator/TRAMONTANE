"""Tramontane — Mistral-native agent orchestration framework."""

from __future__ import annotations

import logging

__version__ = "0.1.3"
__author__ = "Bleucommerce SAS"
__license__ = "MIT"

# Library best practice: let the user configure logging.
logging.getLogger("tramontane").addHandler(logging.NullHandler())
