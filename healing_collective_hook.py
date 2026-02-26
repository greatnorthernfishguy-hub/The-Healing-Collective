"""
The Healing Collective OpenClaw Hook — E-T Systems Standard Integration

Exposes The Healing Collective's self-healing intelligence as an OpenClaw
skill, using the standardized OpenClawAdapter base class.

OpenClaw calls get_instance().on_message(text) on every turn.
The adapter handles all ecosystem wiring (Tier 1/2/3 learning) and
memory logging.  This file implements what's unique to The Healing
Collective:

  - _embed():              Sentence-transformer / hash fallback
  - _module_on_message():  Scan for failure indicators, route to engine
  - _module_stats():       Healing-specific telemetry

Additional public API:
  - report_failure():      Channel 2 host API — explicit failure report
  - get_healing_status():  Query status of a reported failure
  - register_primitive():  Register custom repair primitive

SKILL.md entry:
    name: healing_collective
    autoload: true
    hook: healing_collective_hook.py::get_instance

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: HealingCollectiveHook — OpenClawAdapter subclass with Diagnosis
#         Engine, DVS, and repair primitive registry.
#   Why:  PRD §4.2 specifies this as the primary entry point for the module.
#         Follows the canonical pattern from trollguard_hook.py.
#   Settings: All configuration loaded from config.yaml via core/config.py.
#   How:  Subclass OpenClawAdapter, initialize DVS + DiagnosisEngine in
#         __init__, scan messages for failure indicators, route to engine.
#         Singleton via get_instance().
# -------------------
"""

from __future__ import annotations

import logging
import os
import re
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

logger = logging.getLogger("healing_collective_hook")

# Failure indicator patterns for message scanning
_FAILURE_PATTERNS = [
    re.compile(r"\b(error|exception|traceback|failed|failure|crash|fatal)\b", re.IGNORECASE),
    re.compile(r"\b(timeout|timed\s*out|connection\s*refused|unreachable)\b", re.IGNORECASE),
    re.compile(r"\b(oom|out\s*of\s*memory|memory\s*error|segfault)\b", re.IGNORECASE),
    re.compile(r"\b(permission\s*denied|access\s*denied|unauthorized)\b", re.IGNORECASE),
    re.compile(r"\b(corrupt|corrupted|inconsistent|integrity)\b", re.IGNORECASE),
]


class HealingCollectiveHook(OpenClawAdapter):
    """OpenClaw integration hook for The Healing Collective.

    Primary entry point for the module.  Subclasses OpenClawAdapter
    following the canonical pattern from trollguard_hook.py.
    """

    MODULE_ID = "healing_collective"
    SKILL_NAME = "Healing Collective"
    WORKSPACE_ENV = "HEALING_COLLECTIVE_WORKSPACE_DIR"
    DEFAULT_WORKSPACE = "~/.openclaw/healing_collective"

    def __init__(self) -> None:
        super().__init__()

        # Import core modules (after OpenClawAdapter init sets up ecosystem)
        from core.config import HealingCollectiveConfig
        from core.diagnosis_engine import DiagnosisEngine
        from core.dvs import DiagnosticVectorStore
        from core.repair_primitives import DEFAULT_PRIMITIVES, RepairPrimitive

        # Load configuration
        self._config = HealingCollectiveConfig.from_yaml()

        # Module data directory
        self._module_dir = Path.home() / ".et_modules" / "healing_collective"
        self._module_dir.mkdir(parents=True, exist_ok=True)
        (self._module_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize DVS with NG-Lite substrate from ecosystem
        dvs_path = str(self._module_dir / "dvs.msgpack")
        ng_lite = self._eco._ng if self._eco and hasattr(self._eco, "_ng") else None
        self._dvs = DiagnosticVectorStore(
            max_entries=self._config.dvs_max_entries,
            persistence_path=dvs_path,
            ng_lite=ng_lite,
        )

        # Initialize repair primitives registry
        self._primitives: Dict[str, RepairPrimitive] = dict(DEFAULT_PRIMITIVES)

        # Initialize Diagnosis Engine
        self._engine = DiagnosisEngine(
            config=self._config,
            dvs=self._dvs,
            ng_ecosystem=self._eco,
            primitives=self._primitives,
            embed_fn=self._embed,
        )

        # Checkpoint timer
        self._checkpoint_timer: Optional[threading.Timer] = None
        self._schedule_checkpoint()

        # Signal handlers for graceful shutdown
        self._register_shutdown_handlers()

        logger.info(
            "[%s] Healing Collective ready (tier %d, %d DVS entries, %d primitives)",
            self.MODULE_ID,
            self._eco.tier if self._eco else 0,
            self._dvs.size,
            len(self._primitives),
        )

    # -----------------------------------------------------------------
    # OpenClawAdapter implementation
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using sentence-transformers, fall back to hash."""
        try:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, "_st_model"):
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return np.array(vec, dtype=np.float32)
        except Exception:
            return self._hash_embed(text)

    def _module_on_message(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Scan message for failure indicators and route to engine.

        The Collective monitors the OpenClaw message stream for failure
        indicators.  Messages containing failure signals are routed to
        the Diagnosis Engine.  All messages contribute to the baseline.
        """
        result: Dict[str, Any] = {}

        # Scan for failure indicators
        failure_detected = False
        matched_patterns = []
        for pattern in _FAILURE_PATTERNS:
            match = pattern.search(text)
            if match:
                failure_detected = True
                matched_patterns.append(match.group())

        result["failure_detected"] = failure_detected

        if failure_detected:
            # Route to Diagnosis Engine
            try:
                diagnosis = self._engine.diagnose(
                    description=text,
                    metadata={"matched_patterns": matched_patterns, "source": "openclaw_message"},
                    source="host",
                )
                result["diagnosis"] = {
                    "tracking_id": diagnosis.tracking_id,
                    "novelty": diagnosis.novelty,
                    "proposed_primitive": diagnosis.proposed_primitive,
                    "confidence": diagnosis.confidence,
                    "action_taken": diagnosis.action_taken,
                }
            except Exception as exc:
                logger.warning("Diagnosis failed for message: %s", exc)
                result["diagnosis_error"] = str(exc)

        return result

    def _module_stats(self) -> Dict[str, Any]:
        """Healing Collective-specific telemetry."""
        engine_stats = self._engine.stats()
        dvs_stats = self._dvs.stats()

        return {
            "failures_observed": engine_stats["failures_observed"],
            "repairs_executed": engine_stats["repairs_executed"],
            "repairs_succeeded": engine_stats["repairs_succeeded"],
            "repair_success_rate": engine_stats["repair_success_rate"],
            "dvs_entries": dvs_stats["total_entries"],
            "dvs_fullness_pct": dvs_stats["fullness_pct"],
            "substrate_augmented": dvs_stats["substrate_augmented"],
            "primitives_registered": engine_stats["primitives_registered"],
            "active_cooldowns": engine_stats["active_cooldowns"],
        }

    # -----------------------------------------------------------------
    # Public API: Channel 2 Host API
    # -----------------------------------------------------------------

    def report_failure(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Report a failure explicitly from the host application.

        Channel 2 intake per PRD §2.2.2.  This is the primary intake
        for host-system failures not visible in the peer bridge.

        Args:
            description: Human-readable failure description.
            metadata: Additional context (stack traces, metrics, etc.).

        Returns:
            Tracking UUID for status queries.
        """
        result = self._engine.diagnose(
            description=description,
            metadata=metadata,
            source="host",
        )
        return result.tracking_id

    def get_healing_status(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """Query status of a reported failure by tracking ID.

        Args:
            tracking_id: UUID returned by report_failure().

        Returns:
            Status dict or None if tracking_id not found.
        """
        return self._engine.get_status(tracking_id)

    def register_primitive(self, name: str, instance: Any) -> None:
        """Register a custom repair primitive.

        The primitive MUST implement the RepairPrimitive ABC
        (validate()/execute() methods).

        Args:
            name: Snake_case primitive name.
            instance: RepairPrimitive instance.
        """
        from core.repair_primitives import RepairPrimitive

        if not isinstance(instance, RepairPrimitive):
            raise TypeError(
                f"Primitive must be a RepairPrimitive subclass, got {type(instance)}"
            )
        self._primitives[name] = instance
        self._engine.register_primitive(name, instance)

    # -----------------------------------------------------------------
    # Checkpointing (PRD §6.2)
    # -----------------------------------------------------------------

    def _schedule_checkpoint(self) -> None:
        """Schedule periodic checkpoint."""
        interval = self._config.checkpoint_interval_seconds
        self._checkpoint_timer = threading.Timer(interval, self._do_checkpoint)
        self._checkpoint_timer.daemon = True
        self._checkpoint_timer.start()

    def _do_checkpoint(self) -> None:
        """Execute checkpoint: save DVS + NG-Lite state atomically."""
        try:
            self._dvs.save()
            if self._eco:
                self._eco.save()
            logger.debug("Checkpoint completed")
        except Exception as exc:
            logger.warning("Checkpoint failed: %s", exc)
        finally:
            self._schedule_checkpoint()

    def checkpoint_now(self) -> None:
        """Force an immediate checkpoint (e.g., after successful repair)."""
        try:
            self._dvs.save()
            if self._eco:
                self._eco.save()
        except Exception as exc:
            logger.warning("Manual checkpoint failed: %s", exc)

    def _register_shutdown_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown."""
        def _shutdown_handler(signum, frame):
            logger.info("Shutdown signal received, saving state...")
            try:
                self._dvs.save()
                if self._eco:
                    self._eco.save()
            except Exception as exc:
                logger.warning("Shutdown save failed: %s", exc)

        try:
            signal.signal(signal.SIGTERM, _shutdown_handler)
            signal.signal(signal.SIGINT, _shutdown_handler)
        except Exception:
            # May fail if not in main thread
            pass

    # -----------------------------------------------------------------
    # Checkpoint export/import (PRD §6.3)
    # -----------------------------------------------------------------

    def export_checkpoint(self, export_dir: Optional[str] = None) -> str:
        """Export a checkpoint archive for bootstrapping fresh deployments.

        Args:
            export_dir: Directory to write the archive.  Defaults to
                        checkpoints/ subdirectory.

        Returns:
            Path to the exported checkpoint archive.
        """
        if export_dir is None:
            export_dir = str(self._module_dir / "checkpoints")

        os.makedirs(export_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(export_dir, f"checkpoint_{timestamp}")
        os.makedirs(archive_dir)

        # Save current state
        self._dvs.save(os.path.join(archive_dir, "dvs.msgpack"))
        if self._eco:
            self._eco.save()
            state_path = self._module_dir / "ng_lite_state.json"
            if state_path.exists():
                import shutil
                shutil.copy2(state_path, os.path.join(archive_dir, "ng_lite_state.json"))

        logger.info("Checkpoint exported to %s", archive_dir)
        return archive_dir

    def import_checkpoint(self, path: str) -> None:
        """Import a checkpoint archive, replacing current state.

        Args:
            path: Path to checkpoint directory containing dvs.msgpack
                  and optionally ng_lite_state.json.
        """
        from core.dvs import DiagnosticVectorStore

        dvs_file = os.path.join(path, "dvs.msgpack")
        state_file = os.path.join(path, "ng_lite_state.json")

        if os.path.exists(dvs_file):
            ng_lite = self._eco._ng if self._eco and hasattr(self._eco, "_ng") else None
            self._dvs = DiagnosticVectorStore(
                max_entries=self._config.dvs_max_entries,
                persistence_path=str(self._module_dir / "dvs.msgpack"),
                ng_lite=ng_lite,
            )
            self._dvs._load(dvs_file)
            self._dvs.save()
            self._engine._dvs = self._dvs

        if os.path.exists(state_file) and self._eco and self._eco._ng:
            self._eco._ng.load(state_file)
            self._eco.save()

        logger.info("Checkpoint imported from %s", path)


# --------------------------------------------------------------------------
# Singleton wiring — identical pattern for all E-T Systems modules
# --------------------------------------------------------------------------

_INSTANCE: Optional[HealingCollectiveHook] = None


def get_instance() -> HealingCollectiveHook:
    """Return the Healing Collective OpenClaw hook singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HealingCollectiveHook()
    return _INSTANCE
