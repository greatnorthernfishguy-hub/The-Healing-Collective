"""
The Healing Collective — Core Package

Adaptive self-healing intelligence for the E-T Systems ecosystem.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: Core package init exposing primary classes.
#   Why:  Clean imports for healing_collective_hook.py and tests.
#   How:  Re-exports from submodules.
# -------------------
"""

from core.config import HealingCollectiveConfig
from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType
from core.repair_primitives import (
    RepairPrimitive,
    ValidationResult,
    ExecutionResult,
)

__all__ = [
    "HealingCollectiveConfig",
    "DiagnosticVectorStore",
    "DVSEntry",
    "DVSEntryType",
    "RepairPrimitive",
    "ValidationResult",
    "ExecutionResult",
]
