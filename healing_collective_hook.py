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
# [2026-06-22] Claude Code (Opus 4.8) — Migrate THC onto THE COMMONS (substrate-as-protocol, #335)
#   What: THC reconnected to the shared substrate three ways, mirroring QG #325:
#         (1) DEPOSIT — DiagnosisEngine now gets a CommonsEco(namespaces=("repair:",)) instead of
#             ng_ecosystem=None (SKIP_ECOSYSTEM dead-eco). Step-7 record_outcome deposits repair
#             outcomes into the Commons (was a no-op — _eco was None).
#         (2) BUCKET NOVELTY — _bucket_commons_novelty() reads NG's metrics:neurograph:* deposits
#             from the Commons → _substrate_novelty (EWMA). Replaces the dead tract path: the
#             NG→THC tract froze 0-byte on 2026-06-07 throttle, so _on_river_events' ENTRY_TOPOLOGY
#             novelty has been a dead constant 1.0 for weeks.
#         (3) BUCKET EXPERIENCE — _bucket_commons_experience() buckets NG's experience:* raw turns,
#             embeds at THC's OWN boundary (self._embed, LAW 7), and runs _check_failure_from_river.
#             This restores THC's conversation-triggered failure detection, which rode the same dead
#             tract — THC has been DARK (never triggered except via host API). "Actually DO something."
#   Why: Substrate axiom — THC dips its bucket into the shared Commons; nobody pushes to THC. The
#         tract drain (_drain_river/_on_river_events) is the LAW-1 mechanism the Commons replaces.
#   How: _pulse_cycle() calls the two buckets after the (now-empty) _drain_river(). Dedup via
#         _commons_seen (metrics:/experience:/repair: prefixes don't collide). Fail-soft throughout —
#         no Commons (standalone/Tier-1) → buckets no-op, CommonsEco→None. _on_river_events left intact
#         (harmless: drains empty events). health_monitor/congregation/tier3 still ng_ecosystem=None —
#         they reach into _eco._ng/_eco._peer_bridge private internals, a separate migration (FLAGGED).
# [2026-05-25] Claude Code (Sonnet 4.6) — Wire substrate novelty from River (NeuroGraph proper)
#   What: _on_river_events() now reads ENTRY_TOPOLOGY BTF events from NG proper's tract and
#         extracts predictions_surprised/(confirmed+surprised) as the live novelty ratio.
#         EWMA (α=0.2) smooths across step cycles. DiagnosisEngine receives substrate_novelty
#         via context instead of hardcoded 1.0. Bootstraps at 1.0 (maximum caution) until
#         NG proper's step data arrives.
#   Why:  novelty=1.0 hardcoded treated every failure as completely unknown forever, suppressing
#         the confidence ceiling even for patterns the SNN has learned well. The SNN decides
#         novelty natively — predictions_surprised is the substrate's own measure of surprise.
#         Same epistemic principle as competence graduation: earned trust, not granted.
#   How:  _substrate_novelty float on hook, updated in _on_river_events() per BTF topology
#         event. Both diagnose() call sites pass context={"substrate_novelty": self._substrate_novelty}.
# [2026-04-19] Claude Code — #5: replace dead eco drain with _drain_river() + _on_river_events()
#   What: _pulse_cycle() now calls _drain_river(); failure routing moved to _on_river_events() override
#   Why: #5 — eco._peer_bridge was dead (SKIP_ECOSYSTEM); BTF drain is in openclaw_adapter base class
#   How: _on_river_events() receives new BTF events; routes conversation events to _check_failure_from_river()
# [2026-03-28] Claude (Opus 4.6) — Add autonomic pulse loop (#109)
# What: Added _pulse_loop() daemon thread following the Tonic pattern.
#   Added _shutdown_event, _in_conversation flag, dual-interval support,
#   on_conversation_started/ended() methods.
# Why: #109 — Modules must be alive between conversations. THC was only
#   active during fan-out. The pulse loop drains River tracts for peer
#   repair knowledge and syncs cluster knowledge continuously.
# How: _pulse_loop() runs as a daemon thread started at the end of
#   __init__. Each cycle drains peer bridge tracts and syncs tier3
#   cluster knowledge. Resting interval 30s, conversation interval 10s.
#   Existing health monitor, compression, and checkpoint threads unchanged.
# -------------------
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to BAAI/bge-base-en-v1.5 (#45)
# What: fastembed model all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5 (768-dim).
# Why: Ecosystem-wide embedding migration. Punchlist #45.
# How: TextEmbedding() model string + docstring update.
# -------------------
# [2026-03-18] Claude (CC) — Replace regex failure gate with substrate detection
# What: Removed _FAILURE_PATTERNS regex list and keyword-matching gate.
#   Messages are now routed to the diagnosis engine based on substrate
#   signals: DVS similarity to known failure signatures and/or high
#   substrate novelty (unknown-but-suspicious patterns). Detection
#   thresholds are managed by DetectionCalibrator with three-tier
#   competence model (Apprentice → Journeyman → Master).
# Why: Punch list #70 — Law 7 violation. The regex patterns pre-classified
#   messages before the substrate could see them. The substrate should
#   make this call, with increasing autonomy as it demonstrates competence.
# How: _module_on_message() probes DVS for similarity and checks substrate
#   novelty. DetectionCalibrator provides thresholds — static in Apprentice
#   mode, bounded-adaptive in Journeyman, unbounded in Master. Graduation
#   is competence-based: outcome count + accuracy, not time. Calibrator
#   state persists to detection_calibrator.json.
# -------------------
# [2026-02-27] Claude (Opus 4.6) — Phase 3+4 integration.
#   What: Added Health Monitor, Congregation, Compression, and Tier 3
#         Coordinator initialization.  Updated _module_stats() with
#         Phase 3+4 telemetry.  Added compression timer.
#   Why:  Phases 3+4 complete the Alpha feature set.
#   How:  New modules initialized after engine to break circular deps.
#         Health monitor and compression timer run as daemon threads.
#
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

# Auto-update on startup — pull latest code + sync vendored files
try:
    from ng_updater import auto_update; auto_update()
except Exception:
    pass  # Never prevent module startup

import logging
import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

try:
    from ng_commons_eco import CommonsEco   # vendored Commons-backed eco adapter (#335)
except Exception:  # noqa: BLE001 — standalone/Tier-1 without the Commons toolkit on path
    CommonsEco = None

logger = logging.getLogger("healing_collective_hook")


class HealingCollectiveHook(OpenClawAdapter):
    """OpenClaw integration hook for The Healing Collective.

    Primary entry point for the module.  Subclasses OpenClawAdapter
    following the canonical pattern from trollguard_hook.py.
    """

    MODULE_ID = "healing_collective"
    SKIP_ECOSYSTEM = True
    SKILL_NAME = "Healing Collective"
    WORKSPACE_ENV = "HEALING_COLLECTIVE_WORKSPACE_DIR"
    DEFAULT_WORKSPACE = "~/.openclaw/healing_collective"

    def __init__(self) -> None:
        super().__init__()

        # Import core modules (after OpenClawAdapter init sets up ecosystem)
        from core.config import HealingCollectiveConfig
        from core.detection_calibrator import DetectionCalibrator
        from core.diagnosis_engine import DiagnosisEngine
        from core.dvs import DiagnosticVectorStore, DVSEntryType
        from core.repair_primitives import DEFAULT_PRIMITIVES, RepairPrimitive

        # Cache the entry type enum for substrate-based failure detection
        self._dvs_failure_type = DVSEntryType.FAILURE_SIGNATURE

        # Load configuration
        self._config = HealingCollectiveConfig.from_yaml()

        # Module data directory
        self._module_dir = Path.home() / ".et_modules" / "healing_collective"
        self._module_dir.mkdir(parents=True, exist_ok=True)
        (self._module_dir / "checkpoints").mkdir(exist_ok=True)

        # Detection calibrator — three-tier competence model for
        # failure detection thresholds (Apprentice → Journeyman → Master)
        self._calibrator = DetectionCalibrator(
            persistence_path=str(self._module_dir / "detection_calibrator.json"),
        )

        # Initialize DVS with NG-Lite substrate from ecosystem
        dvs_path = str(self._module_dir / "dvs.msgpack")
        ng_lite = None  # local graph removed — reads come from topology delta
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
            # #335: Commons-backed eco (was None under SKIP_ECOSYSTEM). Step-7 record_outcome
            # deposits repair outcomes to the shared Commons under the "repair:" namespace.
            ng_ecosystem=(CommonsEco(namespaces=("repair:",), source_id="healing_collective")
                          if CommonsEco else None),
            primitives=self._primitives,
            embed_fn=self._embed,
        )

        # --- Phase 3: Health Monitor + Congregation ---
        from core.health_monitor import HealthMonitor
        from core.congregation import Congregation

        self._health_monitor = HealthMonitor(
            config=self._config.health_monitor,
            ng_ecosystem=None,
            dvs=self._dvs,
            engine=self._engine,
        )

        self._congregation = Congregation(
            config=self._config.congregation,
            ng_ecosystem=None,
            dvs=self._dvs,
            embed_fn=self._embed,
        )
        self._engine._congregation = self._congregation

        # --- Phase 4: Compression + Tier 3 Coordinator ---
        from core.compression import PatternCompressor
        from core.tier3_upgrade import Tier3Coordinator

        self._compressor = PatternCompressor(
            config=self._config.compression,
            dvs=self._dvs,
        )

        self._tier3 = Tier3Coordinator(
            module_id=self.MODULE_ID,
            ng_ecosystem=None,
            dvs=self._dvs,
            embed_fn=self._embed,
        )
        self._engine._tier3 = self._tier3

        # Startup: sync cluster knowledge (Tier 3)
        try:
            self._tier3.sync_cluster_knowledge()
        except Exception:
            pass

        # Start Health Monitor background thread
        self._health_monitor.start()

        # Checkpoint timer
        self._checkpoint_timer: Optional[threading.Timer] = None
        self._schedule_checkpoint()

        # Compression timer (runs once per cycle)
        self._compression_timer: Optional[threading.Timer] = None
        self._schedule_compression()

        # Signal handlers for graceful shutdown
        self._register_shutdown_handlers()

        # Substrate novelty from NeuroGraph proper's River (EWMA, bootstraps at 1.0)
        self._substrate_novelty: float = 1.0

        # #335: dedup set for bucketed Commons deposits (metrics:/experience: prefixes don't collide).
        # Novelty EWMA + failure detection must not re-process the same deposit across pulses.
        self._commons_seen: set = set()

        # --- Pulse loop (#109) ---
        self._shutdown_event = threading.Event()
        self._in_conversation = False
        self._resting_interval = 30.0
        self._conversation_interval = 10.0

        self._pulse_thread = threading.Thread(
            target=self._pulse_loop,
            name="thc-pulse",
            daemon=True,
        )
        self._pulse_thread.start()

        logger.info(
            "[%s] Healing Collective ready (tier %d, %d DVS entries, %d primitives, "
            "health_monitor=%s, congregation=%s, compression=%s, tier3=%s)",
            self.MODULE_ID,
            0,  # tier reported via tracts
            self._dvs.size,
            len(self._primitives),
            "on" if self._config.health_monitor.enabled else "off",
            "on",
            "on",
            "on",
        )

    # -----------------------------------------------------------------
    # OpenClawAdapter implementation
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text via ng_embed (centralized ecosystem embedding).

        Ecosystem standard: Snowflake/snowflake-arctic-embed-m-v1.5 (768-dim).
        ONNX Runtime, no torch dependency.
        """
        try:
            from ng_embed import embed
            return embed(text)
        except Exception:
            return self._hash_embed(text)

    def _module_on_message(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """No-op — failure detection runs from River events in pulse cycle.

        Conversation content arrives via topology delta. The pulse cycle
        extracts text+embedding and runs DVS similarity + novelty checks.
        """
        return {}

    def _module_stats(self) -> Dict[str, Any]:
        """Healing Collective-specific telemetry."""
        engine_stats = self._engine.stats()
        dvs_stats = self._dvs.stats()

        stats = {
            "failures_observed": engine_stats["failures_observed"],
            "repairs_executed": engine_stats["repairs_executed"],
            "repairs_succeeded": engine_stats["repairs_succeeded"],
            "repair_success_rate": engine_stats["repair_success_rate"],
            "dvs_entries": dvs_stats["total_entries"],
            "dvs_fullness_pct": dvs_stats["fullness_pct"],
            "substrate_augmented": dvs_stats["substrate_augmented"],
            "primitives_registered": engine_stats["primitives_registered"],
            "active_cooldowns": engine_stats["active_cooldowns"],
            "detection_calibrator": self._calibrator.stats(),
            "health_monitor": self._health_monitor.stats(),
            "congregation": self._congregation.stats(),
            "compression": self._compressor.stats(),
            "tier3": self._tier3.stats(),
        }
        return stats

    # -----------------------------------------------------------------
    # Pulse loop (#109) — autonomous between-conversation processing
    # -----------------------------------------------------------------

    def _pulse_loop(self) -> None:
        """Continuous autonomic pulse — the organ stays alive between conversations.

        Follows the Tonic pattern (tonic_engine.py _generation_loop).
        Each cycle drains River tracts for peer repair knowledge and
        syncs cluster knowledge from the tier3 coordinator.
        """
        while not self._shutdown_event.is_set():
            try:
                self._pulse_cycle()
            except Exception as exc:
                logger.debug("THC pulse cycle error: %s", exc)
            interval = (
                self._conversation_interval
                if self._in_conversation
                else self._resting_interval
            )
            self._shutdown_event.wait(timeout=interval)

    def _pulse_cycle(self) -> None:
        """Single pulse cycle — drain River tracts, check for failures, sync clusters."""
        # Drain River tracts via BTF bridge (#5) — legacy path, empty since the 2026-06-07 throttle.
        self._drain_river()
        # #335: the live Commons buckets — THC dips into the shared substrate (substrate axiom).
        self._bucket_commons_novelty()       # NG substrate metrics → _substrate_novelty (EWMA)
        self._bucket_commons_experience()    # NG raw conversation → failure detection (THC's trigger)

    def _bucket_commons_novelty(self) -> None:
        """Bucket NG's substrate metrics from the Commons → THC's novelty signal (#335).

        Replaces the dead tract-drain novelty (_on_river_events ENTRY_TOPOLOGY), frozen since the
        2026-06-07 throttle stopped NG depositing topology to THC's inbound tract. Same EWMA math,
        now fed from the shared Commons. Substrate axiom — THC dips its bucket; nobody pushes.
        """
        try:
            from commons import get_commons
            commons = get_commons()
        except Exception:  # noqa: BLE001 — no Commons (standalone) → nothing to bucket
            return
        if commons is None:
            return
        try:
            recs = commons.bucket_recent(limit=50, with_metadata=True)
        except Exception as exc:  # noqa: BLE001 — a bucket failure never breaks the pulse
            logger.debug("THC Commons novelty bucket failed: %s", exc)
            return
        for target_id, _w, _r, meta in recs:
            if not target_id.startswith("metrics:neurograph:") or target_id in self._commons_seen:
                continue
            self._commons_seen.add(target_id)
            surprise = self._surprise_from_metric(meta)
            if surprise is not None:
                self._substrate_novelty = (
                    0.8 * self._substrate_novelty + 0.2 * surprise
                )
        if len(self._commons_seen) > 4096:           # bound the dedup set
            self._commons_seen = set(list(self._commons_seen)[-2048:])

    def _bucket_commons_experience(self) -> None:
        """Bucket NG's raw conversation experience from the Commons → THC failure detection (#335).

        THC's conversation-triggered failure detection rode the dead tract too (THC has been DARK,
        triggered only via the host report_failure API). Restore it: bucket experience:* raw turns,
        embed at THC's OWN extraction boundary (self._embed, LAW 7 — classify/embed at the bucket),
        run _check_failure_from_river. Dedup via _commons_seen. Fail-soft.
        """
        try:
            from commons import get_commons
            commons = get_commons()
        except Exception:  # noqa: BLE001 — no Commons (standalone) → nothing to bucket
            return
        if commons is None:
            return
        try:
            recs = commons.bucket_recent(limit=50, with_metadata=True)
        except Exception as exc:  # noqa: BLE001 — a bucket failure never breaks the pulse
            logger.debug("THC Commons experience bucket failed: %s", exc)
            return
        for target_id, _w, _r, meta in recs:
            if not target_id.startswith("experience:") or target_id in self._commons_seen:
                continue
            self._commons_seen.add(target_id)
            if not isinstance(meta, dict):
                continue
            text = "\n\n".join(p for p in (meta.get("user_text"), meta.get("assistant_text")) if p)
            if not text:
                continue
            try:
                emb = self._embed(text)          # THC embeds at its OWN boundary (LAW 7)
                if emb is None:
                    continue
                self._check_failure_from_river({"text": text, "embedding": emb})
            except Exception as exc:  # noqa: BLE001 — one bad turn never breaks the pulse
                logger.debug("THC failure-check from Commons experience failed: %s", exc)
        if len(self._commons_seen) > 4096:
            self._commons_seen = set(list(self._commons_seen)[-2048:])

    @staticmethod
    def _surprise_from_metric(meta) -> "Optional[float]":
        """Extract NG's surprise ratio from a bucketed metric deposit (anomaly or nominal span)."""
        if not isinstance(meta, dict):
            return None
        if meta.get("salience") == "anomaly" and "signal" in meta:
            return float(meta["signal"])            # the surprise ratio, directly
        agg = meta.get("aggregate") if meta.get("salience") == "nominal" else None
        if isinstance(agg, dict):
            c = agg.get("predictions_confirmed", 0)
            s = agg.get("predictions_surprised", 0)
            t = c + s
            return (s / t) if t else 0.0            # nominal span ⇒ low/zero surprise (still informative)
        return None

    def _on_river_events(self, events: list) -> None:
        """Route new River events through failure-detection bucket."""
        # Extract substrate novelty from NeuroGraph proper's topology deposits
        try:
            import ng_tract
            import msgpack
            for entry in events:
                if not hasattr(entry, "entry_type"):
                    continue
                if entry.entry_type != ng_tract.ENTRY_TOPOLOGY:
                    continue
                try:
                    payload = msgpack.unpackb(entry.raw(), raw=False)
                    confirmed = payload.get("predictions_confirmed", 0)
                    surprised = payload.get("predictions_surprised", 0)
                    total = confirmed + surprised
                    if total > 0:
                        step_novelty = surprised / total
                        self._substrate_novelty = (
                            0.8 * self._substrate_novelty + 0.2 * step_novelty
                        )
                except Exception:
                    pass
        except ImportError:
            pass

        for event in events:
            if isinstance(event, dict) and event.get("conversation"):
                try:
                    self._check_failure_from_river(event["conversation"])
                except Exception as exc:
                    logger.debug("Pulse failure check error: %s", exc)

        # Sync cluster knowledge from tier3 coordinator
        try:
            self._tier3.sync_cluster_knowledge()
        except Exception as exc:
            logger.debug("Pulse tier3 sync failed: %s", exc)

    def _check_failure_from_river(self, conversation: dict) -> None:
        """Check conversation content from River for failure signals.

        Same logic as the old _module_on_message: DVS similarity check,
        novelty detection, route to diagnosis engine if triggered.
        """
        text = conversation.get("text", "")
        raw_emb = conversation.get("embedding")
        if not text or raw_emb is None:
            return

        import numpy as _np
        embedding = _np.asarray(raw_emb, dtype=_np.float32)

        # Probe DVS for similarity to known failure signatures
        dvs_similarity = 0.0
        try:
            dvs_hits = self._dvs.search(embedding, top_k=1, entry_type=self._dvs_failure_type)
            if dvs_hits:
                dvs_similarity = dvs_hits[0][1]
        except Exception:
            pass

        # Substrate novelty from NeuroGraph proper (EWMA updated in _on_river_events)
        novelty = self._substrate_novelty

        # Get adaptive thresholds
        sim_threshold, nov_threshold = self._calibrator.get_thresholds()
        similarity_triggered = dvs_similarity >= sim_threshold
        novelty_triggered = novelty >= nov_threshold

        if not (similarity_triggered or novelty_triggered):
            return

        trigger = "dvs_similarity" if similarity_triggered else "novelty"
        try:
            diagnosis = self._engine.diagnose(
                description=text,
                metadata={
                    "source": "river_conversation",
                    "dvs_similarity": dvs_similarity,
                    "novelty": novelty,
                    "trigger": trigger,
                    "detection_tier": self._calibrator.tier.value,
                },
                source="host",
                context={"substrate_novelty": self._substrate_novelty},
            )
            was_real = (
                diagnosis.proposed_primitive is not None
                and diagnosis.confidence >= self._config.confidence_recommend
            )
            self._calibrator.record_outcome(
                similarity_score=dvs_similarity,
                novelty_score=novelty,
                trigger=trigger,
                was_real_failure=was_real,
            )
        except Exception as exc:
            logger.debug("River failure diagnosis error: %s", exc)

    def on_conversation_started(self) -> None:
        """Mode swap: conversation active — shorter pulse interval."""
        self._in_conversation = True

    def on_conversation_ended(self) -> None:
        """Mode swap: conversation ended — longer pulse interval."""
        self._in_conversation = False

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
            context={"substrate_novelty": self._substrate_novelty},
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
    # Compression timer (Phase 4)
    # -----------------------------------------------------------------

    def _schedule_compression(self) -> None:
        """Schedule periodic compression check."""
        # Check every hour whether compression should run
        self._compression_timer = threading.Timer(3600, self._do_compression)
        self._compression_timer.daemon = True
        self._compression_timer.start()

    def _do_compression(self) -> None:
        """Check and run compression if needed."""
        try:
            if self._compressor.should_compress():
                self._compressor.compress()
        except Exception as exc:
            logger.warning("Compression cycle failed: %s", exc)
        finally:
            self._schedule_compression()

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
                pass  # state via tracts
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
                pass  # state via tracts
        except Exception as exc:
            logger.warning("Manual checkpoint failed: %s", exc)

    def _register_shutdown_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers for graceful shutdown."""
        def _shutdown_handler(signum, frame):
            logger.info("Shutdown signal received, saving state...")
            try:
                self._shutdown_event.set()
                self._health_monitor.stop()
                if self._compression_timer:
                    self._compression_timer.cancel()
                self._dvs.save()
                if self._eco:
                    pass  # state via tracts
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
            ng_lite = None  # local graph removed — reads come from topology delta
            self._dvs = DiagnosticVectorStore(
                max_entries=self._config.dvs_max_entries,
                persistence_path=str(self._module_dir / "dvs.msgpack"),
                ng_lite=ng_lite,
            )
            self._dvs._load(dvs_file)
            self._dvs.save()
            self._engine._dvs = self._dvs

        pass  # local state removed — Tier 3 reads from central substrate

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
