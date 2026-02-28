"""
The Healing Collective — Diagnosis Engine

The intelligence layer connecting failure observation to repair action.
Leverages the NG-Lite substrate's learned associations, not rule-based
scoring.

Seven-Step Diagnosis Pipeline:
  1. Observe   — embed and store failure as FAILURE_SIGNATURE
  2. Recognize — activate substrate, detect novelty
  3. Recall    — query DVS through substrate topology
  4. Propose   — select repair primitive + parameters
  5. Validate  — call primitive's validate()
  6. Execute   — if confidence >= threshold and validate passed
  7. Learn     — store diagnostic chain, share via peer bridge

ENFORCEMENT: execute() is NEVER called without preceding validate()
returning passed=True.  This is enforced in code, not by convention.

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Phase 3+4 integration.
#   What: Added Congregation deliberation in recommend zone and Tier 3
#         repair broadcast after successful execution.
#   Why:  Congregation elevates uncertain repairs via peer consensus.
#         Tier 3 broadcasts share repair outcomes across the cluster.
#   How:  Optional _congregation and _tier3 collaborators set post-init.
#         Congregation consulted between Propose and Validate when action
#         is "recommend".  Tier 3 broadcast fires after successful Execute.
#
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: DiagnosisEngine implementing the 7-step pipeline from PRD §2.2.5.
#   Why:  The engine connects failure observation to repair action using
#         the NG-Lite substrate's learned associations.
#   Settings: confidence_auto_execute=0.70, confidence_recommend=0.40,
#         confidence_host_premium=0.15, repair_cooldown_seconds=300.
#   How:  Pipeline processes failures through observe→recognize→recall→
#         propose→validate→execute→learn.  Confidence thresholds determine
#         whether to auto-execute, recommend, or silently log.
# -------------------
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import HealingCollectiveConfig
from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType
from core.repair_primitives import (
    ExecutionResult,
    RepairPrimitive,
    ValidationResult,
)

logger = logging.getLogger("healing_collective.diagnosis_engine")


# ---------------------------------------------------------------------------
# Diagnosis result types
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Complete result of a diagnosis pipeline run."""
    tracking_id: str
    failure_description: str
    novelty: float
    proposed_primitive: Optional[str] = None
    confidence: float = 0.0
    action_taken: str = "silent_log"  # "auto_execute" | "recommend" | "silent_log"
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[ExecutionResult] = None
    diagnostic_chain: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Diagnosis Engine
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """Seven-step diagnosis pipeline.

    Connects failure observation to repair action using the NG-Lite
    substrate's learned associations.
    """

    def __init__(
        self,
        config: HealingCollectiveConfig,
        dvs: DiagnosticVectorStore,
        ng_ecosystem: Any,
        primitives: Dict[str, RepairPrimitive],
        embed_fn: Any,
    ):
        """
        Args:
            config: Healing Collective configuration.
            dvs: Diagnostic Vector Store instance.
            ng_ecosystem: NGEcosystem instance (provides substrate access).
            primitives: Registry of available repair primitives.
            embed_fn: Callable that embeds text to np.ndarray.
        """
        self._config = config
        self._dvs = dvs
        self._eco = ng_ecosystem
        self._primitives = dict(primitives)
        self._embed_fn = embed_fn

        # Optional Phase 3+4 collaborators (set post-init to break circular deps)
        self._congregation: Any = None
        self._tier3: Any = None

        # Cooldown tracking: (failure_hash, primitive_name) -> expiry timestamp
        self._cooldowns: Dict[Tuple[str, str], float] = {}

        # Tracking map: tracking_id -> DiagnosisResult
        self._tracking: Dict[str, DiagnosisResult] = {}

        # Stats
        self._failures_observed = 0
        self._repairs_executed = 0
        self._repairs_succeeded = 0

    def register_primitive(self, name: str, primitive: RepairPrimitive) -> None:
        """Register a custom repair primitive."""
        self._primitives[name] = primitive
        logger.info("Registered custom primitive: %s", name)

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------

    def diagnose(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "host",
        context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisResult:
        """Run the seven-step diagnosis pipeline.

        Args:
            description: Human-readable failure description.
            metadata: Additional context (stack traces, metrics, etc.).
            source: "host" or peer module_id.
            context: System context for repair execution.

        Returns:
            DiagnosisResult with the full diagnostic chain.
        """
        tracking_id = str(uuid.uuid4())
        metadata = metadata or {}
        context = context or {}
        self._failures_observed += 1

        # --- Step 1: Observe ---
        embedding = self._embed_fn(description)
        failure_entry = DVSEntry.create(
            entry_type=DVSEntryType.FAILURE_SIGNATURE,
            source_module=source,
            embedding=embedding,
            content={
                "description": description,
                "metadata": metadata,
                "tracking_id": tracking_id,
            },
        )

        # Link to substrate node
        try:
            if self._eco and self._eco._ng:
                node = self._eco._ng.find_or_create_node(embedding)
                failure_entry.ng_node_id = node.node_id
        except Exception:
            pass

        self._dvs.add(failure_entry)

        # --- Step 2: Recognize ---
        novelty = 1.0
        try:
            if self._eco:
                novelty = self._eco.detect_novelty(embedding)
        except Exception:
            pass

        # Novelty-based confidence ceiling: new failures get lower max confidence
        confidence_ceiling = 1.0 if novelty < 0.5 else max(0.5, 1.0 - novelty * 0.5)

        # --- Step 3: Recall ---
        recalled = self._dvs.search(
            embedding,
            top_k=self._config.dvs_search_top_k,
            entry_type=DVSEntryType.REPAIR_RECORD,
        )

        # --- Step 4: Propose ---
        proposed_primitive, confidence = self._propose_repair(
            recalled, novelty, confidence_ceiling, source,
        )

        # Determine action level
        is_host = source == "host"
        effective_threshold = self._config.confidence_auto_execute
        if is_host:
            effective_threshold += self._config.confidence_host_premium

        if confidence >= effective_threshold:
            action = "auto_execute"
        elif confidence >= self._config.confidence_recommend:
            action = "recommend"
        else:
            action = "silent_log"

        # Check cooldown
        failure_hash = self._hash_failure(description)
        if proposed_primitive:
            cooldown_key = (failure_hash, proposed_primitive)
            if cooldown_key in self._cooldowns:
                if time.time() < self._cooldowns[cooldown_key]:
                    action = "recommend"  # Downgrade to recommendation

        # --- Phase 3: Congregation deliberation ---
        # When in the recommend zone, consult peers for consensus
        if (
            action == "recommend"
            and proposed_primitive
            and self._congregation is not None
            and self._congregation.should_deliberate(
                confidence, source, effective_threshold,
                self._config.confidence_recommend,
            )
        ):
            try:
                cong_result = self._congregation.deliberate(
                    failure_embedding=embedding,
                    proposed_primitive=proposed_primitive,
                    local_confidence=confidence,
                    failure_description=description,
                )
                confidence = cong_result.adjusted_confidence
                # Re-evaluate action with adjusted confidence
                if confidence >= effective_threshold:
                    action = "auto_execute"
            except Exception as exc:
                logger.debug("Congregation deliberation failed: %s", exc)

        result = DiagnosisResult(
            tracking_id=tracking_id,
            failure_description=description,
            novelty=novelty,
            proposed_primitive=proposed_primitive,
            confidence=confidence,
            action_taken=action,
        )

        # --- Step 5: Validate ---
        validation = None
        if proposed_primitive and action == "auto_execute":
            primitive = self._primitives.get(proposed_primitive)
            if primitive:
                try:
                    repair_context = {**context, **metadata, "description": description}
                    validation = primitive.validate(repair_context)
                    result.validation_result = validation

                    if not validation.passed:
                        action = "recommend"
                        result.action_taken = action
                        logger.info(
                            "[%s] Validate failed for %s: %s",
                            tracking_id, proposed_primitive, validation.reason,
                        )

                        # Store validation failure for learning
                        self._dvs.add(DVSEntry.create(
                            entry_type=DVSEntryType.DIAGNOSTIC_LOG,
                            source_module="healing_collective",
                            embedding=embedding,
                            content={
                                "tracking_id": tracking_id,
                                "primitive": proposed_primitive,
                                "validation_failed": True,
                                "reason": validation.reason,
                            },
                            confidence=confidence,
                        ))

                except Exception as exc:
                    logger.warning("[%s] Validate error: %s", tracking_id, exc)
                    action = "recommend"
                    result.action_taken = action

        # --- Step 6: Execute ---
        if action == "auto_execute" and proposed_primitive and validation and validation.passed:
            primitive = self._primitives[proposed_primitive]
            try:
                repair_context = {**context, **metadata, "description": description}
                execution = primitive.execute(repair_context)
                result.execution_result = execution
                self._repairs_executed += 1

                # Record outcome for learning
                success = execution.status == "success"
                if success:
                    self._repairs_succeeded += 1

                try:
                    if self._eco:
                        self._eco.record_outcome(
                            embedding,
                            target_id=f"repair:{proposed_primitive}",
                            success=success,
                            metadata={
                                "tracking_id": tracking_id,
                                "status": execution.status,
                            },
                        )
                except Exception:
                    pass

                # Set cooldown
                cooldown_key = (failure_hash, proposed_primitive)
                self._cooldowns[cooldown_key] = (
                    time.time() + self._config.repair_cooldown_seconds
                )

                # Phase 4: Tier 3 broadcast — share repair outcome with cluster
                if self._tier3 is not None:
                    try:
                        self._tier3.broadcast_repair(
                            failure_description=description,
                            embedding=embedding,
                            proposed_primitive=proposed_primitive,
                            confidence=confidence,
                            outcome=execution.status,
                            tracking_id=tracking_id,
                        )
                    except Exception:
                        pass

                logger.info(
                    "[%s] Executed %s: %s (%s)",
                    tracking_id, proposed_primitive,
                    execution.status, execution.detail,
                )

            except Exception as exc:
                logger.error("[%s] Execute error: %s", tracking_id, exc)
                result.execution_result = ExecutionResult(
                    status="failed",
                    detail=f"Execution error: {exc}",
                )

        elif action == "recommend" and proposed_primitive:
            logger.info(
                "[%s] RECOMMEND: %s (confidence=%.2f) for: %s",
                tracking_id, proposed_primitive, confidence, description,
            )

        # --- Step 7: Learn ---
        diagnostic_chain = {
            "tracking_id": tracking_id,
            "failure_description": description,
            "source": source,
            "novelty": novelty,
            "proposed_primitive": proposed_primitive,
            "confidence": confidence,
            "action_taken": result.action_taken,
            "validation_passed": validation.passed if validation else None,
            "execution_status": (
                result.execution_result.status
                if result.execution_result else None
            ),
            "timestamp": time.time(),
        }
        result.diagnostic_chain = diagnostic_chain

        # Store repair record in DVS
        repair_outcome = None
        if result.execution_result:
            repair_outcome = result.execution_result.status

        self._dvs.add(DVSEntry.create(
            entry_type=DVSEntryType.REPAIR_RECORD,
            source_module="healing_collective",
            embedding=embedding,
            content=diagnostic_chain,
            confidence=confidence,
            repair_outcome=repair_outcome,
        ))

        # Track for status queries
        self._tracking[tracking_id] = result

        return result

    # -----------------------------------------------------------------
    # Proposal logic
    # -----------------------------------------------------------------

    def _propose_repair(
        self,
        recalled: List[Tuple[DVSEntry, float]],
        novelty: float,
        confidence_ceiling: float,
        source: str,
    ) -> Tuple[Optional[str], float]:
        """Propose a repair primitive based on recalled DVS knowledge.

        Returns (primitive_name, confidence) or (None, 0.0).
        """
        if not recalled:
            # No recalled knowledge — fallback to log_and_recommend
            return "log_and_recommend", min(0.3, confidence_ceiling)

        # Tally successful repair primitives from recalled entries
        primitive_scores: Dict[str, List[float]] = {}
        for entry, relevance_score in recalled:
            content = entry.content or {}
            primitive = content.get("proposed_primitive")
            if not primitive:
                continue
            if primitive not in self._primitives:
                continue

            outcome = entry.repair_outcome
            if outcome == "success":
                score = relevance_score * entry.confidence
            elif outcome == "partial":
                score = relevance_score * entry.confidence * 0.5
            else:
                score = 0.0

            if primitive not in primitive_scores:
                primitive_scores[primitive] = []
            primitive_scores[primitive].append(score)

        if not primitive_scores:
            return "log_and_recommend", min(0.3, confidence_ceiling)

        # Pick the primitive with the highest aggregate score
        best_primitive = None
        best_confidence = 0.0

        for primitive, scores in primitive_scores.items():
            # Confidence grows with repetition and consistency
            avg_score = sum(scores) / len(scores)
            count_bonus = min(0.3, len(scores) * 0.03)  # Up to 0.3 bonus for 10+ successes
            confidence = min(avg_score + count_bonus, confidence_ceiling)

            if confidence > best_confidence:
                best_confidence = confidence
                best_primitive = primitive

        if best_primitive is None:
            return "log_and_recommend", min(0.3, confidence_ceiling)

        return best_primitive, best_confidence

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _hash_failure(self, description: str) -> str:
        """Hash a failure description for cooldown tracking."""
        import hashlib
        return hashlib.sha256(description.encode()).hexdigest()[:16]

    def get_status(self, tracking_id: str) -> Optional[Dict[str, Any]]:
        """Query status of a diagnosed failure by tracking ID."""
        result = self._tracking.get(tracking_id)
        if result is None:
            return None
        return {
            "tracking_id": result.tracking_id,
            "failure_description": result.failure_description,
            "novelty": result.novelty,
            "proposed_primitive": result.proposed_primitive,
            "confidence": result.confidence,
            "action_taken": result.action_taken,
            "execution_status": (
                result.execution_result.status
                if result.execution_result else None
            ),
            "timestamp": result.timestamp,
        }

    def stats(self) -> Dict[str, Any]:
        """Return diagnosis engine statistics."""
        # Clean expired cooldowns
        now = time.time()
        self._cooldowns = {
            k: v for k, v in self._cooldowns.items() if v > now
        }

        return {
            "failures_observed": self._failures_observed,
            "repairs_executed": self._repairs_executed,
            "repairs_succeeded": self._repairs_succeeded,
            "repair_success_rate": (
                round(self._repairs_succeeded / self._repairs_executed, 3)
                if self._repairs_executed > 0 else 0.0
            ),
            "active_cooldowns": len(self._cooldowns),
            "primitives_registered": len(self._primitives),
            "tracked_diagnoses": len(self._tracking),
        }
