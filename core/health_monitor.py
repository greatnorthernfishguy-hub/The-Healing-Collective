"""
The Healing Collective — Health Monitor

Continuous background monitoring of the NG-Lite substrate health.
Detects anomalies proactively — weight divergence, dead nodes, novelty
saturation — and triggers preventive repairs before failures propagate.

The Health Monitor is the Collective's immune system: always running,
mostly quiet, occasionally nudging the substrate back toward health.

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
#   What: HealthMonitor with background thread, three health checks
#         (weight divergence, firing rate, novelty saturation), and
#         proactive repair triggering.
#   Why:  PRD specifies active health monitoring between failures.
#         Waiting for failures is reactive; the Health Monitor enables
#         the Collective to detect substrate drift and act before
#         failures surface.
#   Settings: health_monitor.enabled=true, interval_seconds=120,
#         weight_divergence_threshold=2.0, min_firing_rate=0.001,
#         novelty_saturation_threshold=0.95.
#   How:  Daemon thread runs check_health() at config interval.
#         Each check inspects substrate stats and DVS state.
#         Issues above threshold trigger a DiagnosisEngine.diagnose()
#         call with synthetic failure descriptions.
# -------------------
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import HealthMonitorConfig

logger = logging.getLogger("healing_collective.health_monitor")


# ---------------------------------------------------------------------------
# Health check result types
# ---------------------------------------------------------------------------

@dataclass
class HealthIssue:
    """A single detected health issue."""
    category: str  # "weight_divergence" | "low_firing_rate" | "novelty_saturation"
    severity: float  # 0.0 (minor) to 1.0 (critical)
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Result of a complete health check cycle."""
    timestamp: float = field(default_factory=time.time)
    issues: List[HealthIssue] = field(default_factory=list)
    substrate_nodes: int = 0
    substrate_synapses: int = 0
    dvs_fullness_pct: float = 0.0

    @property
    def healthy(self) -> bool:
        return len(self.issues) == 0


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Background substrate health monitor.

    Periodically inspects the NG-Lite substrate and DVS for signs of
    drift or degradation.  When issues are detected, triggers proactive
    repairs through the diagnosis engine.
    """

    def __init__(
        self,
        config: HealthMonitorConfig,
        ng_ecosystem: Any,
        dvs: Any,
        engine: Optional[Any] = None,
    ):
        """
        Args:
            config: Health monitor configuration.
            ng_ecosystem: NGEcosystem instance (provides substrate access).
            dvs: DiagnosticVectorStore instance.
            engine: DiagnosisEngine instance (for triggering proactive repairs).
        """
        self._config = config
        self._eco = ng_ecosystem
        self._dvs = dvs
        self._engine = engine

        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._check_count = 0
        self._issues_detected = 0
        self._last_report: Optional[HealthReport] = None

    def set_engine(self, engine: Any) -> None:
        """Attach the diagnosis engine (resolves circular init order)."""
        self._engine = engine

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring thread."""
        if not self._config.enabled:
            logger.debug("Health monitor disabled by config")
            return
        self._running = True
        self._schedule_next()
        logger.info(
            "Health monitor started (interval=%ds)",
            self._config.interval_seconds,
        )

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("Health monitor stopped")

    def _schedule_next(self) -> None:
        """Schedule the next health check."""
        if not self._running:
            return
        self._timer = threading.Timer(
            self._config.interval_seconds,
            self._run_cycle,
        )
        self._timer.daemon = True
        self._timer.start()

    def _run_cycle(self) -> None:
        """Execute one health check cycle and reschedule."""
        try:
            report = self.check_health()
            self._last_report = report

            if not report.healthy and self._engine is not None:
                for issue in report.issues:
                    self._trigger_proactive_repair(issue)
        except Exception as exc:
            logger.warning("Health check cycle failed: %s", exc)
        finally:
            self._schedule_next()

    # -----------------------------------------------------------------
    # Health checks
    # -----------------------------------------------------------------

    def check_health(self) -> HealthReport:
        """Run all health checks and return a report.

        Can be called directly for on-demand health checks, or
        runs automatically on the background timer.
        """
        self._check_count += 1
        issues: List[HealthIssue] = []

        # Gather substrate stats
        substrate_nodes = 0
        substrate_synapses = 0
        ng_stats: Dict[str, Any] = {}

        if self._eco and self._eco._ng:
            try:
                ng_stats = self._eco._ng.get_stats()
                substrate_nodes = ng_stats.get("node_count", 0)
                substrate_synapses = ng_stats.get("synapse_count", 0)
            except Exception:
                pass

        # DVS fullness
        dvs_fullness = 0.0
        if self._dvs:
            try:
                dvs_stats = self._dvs.stats()
                dvs_fullness = dvs_stats.get("fullness_pct", 0.0)
            except Exception:
                pass

        # Check 1: Weight divergence
        issue = self._check_weight_divergence(ng_stats)
        if issue:
            issues.append(issue)

        # Check 2: Firing rates (dead nodes)
        issue = self._check_firing_rates(ng_stats)
        if issue:
            issues.append(issue)

        # Check 3: Novelty saturation
        issue = self._check_novelty_saturation()
        if issue:
            issues.append(issue)

        self._issues_detected += len(issues)

        report = HealthReport(
            issues=issues,
            substrate_nodes=substrate_nodes,
            substrate_synapses=substrate_synapses,
            dvs_fullness_pct=dvs_fullness,
        )

        if issues:
            logger.info(
                "Health check #%d: %d issue(s) detected",
                self._check_count, len(issues),
            )
        else:
            logger.debug("Health check #%d: healthy", self._check_count)

        return report

    def _check_weight_divergence(self, ng_stats: Dict[str, Any]) -> Optional[HealthIssue]:
        """Check if synapse weights are dangerously diverged.

        Weight divergence indicates the substrate has learned extreme
        associations — some synapses near 1.0, others near 0.0 — which
        reduces its ability to adapt to new patterns.
        """
        if not self._eco or not self._eco._ng:
            return None

        try:
            synapses = self._eco._ng.synapses
            if not synapses:
                return None

            weights = [s.weight for s in synapses.values()]
            if len(weights) < 2:
                return None

            std_dev = float(np.std(weights))
            mean_weight = float(np.mean(weights))

            # Divergence ratio: std_dev relative to mean
            if mean_weight > 0:
                divergence = std_dev / mean_weight
            else:
                divergence = std_dev

            if divergence > self._config.weight_divergence_threshold:
                return HealthIssue(
                    category="weight_divergence",
                    severity=min(1.0, divergence / (self._config.weight_divergence_threshold * 2)),
                    description=(
                        f"Substrate weight divergence ({divergence:.2f}) exceeds "
                        f"threshold ({self._config.weight_divergence_threshold}). "
                        f"Mean={mean_weight:.3f}, StdDev={std_dev:.3f}"
                    ),
                    metadata={
                        "divergence": divergence,
                        "mean_weight": mean_weight,
                        "std_dev": std_dev,
                        "synapse_count": len(weights),
                    },
                )

        except Exception as exc:
            logger.debug("Weight divergence check failed: %s", exc)

        return None

    def _check_firing_rates(self, ng_stats: Dict[str, Any]) -> Optional[HealthIssue]:
        """Check for dead or underactive nodes.

        Nodes that never fire are wasting capacity.  If too many nodes
        have firing rates below min_firing_rate, the substrate is
        carrying dead weight that should be pruned.
        """
        if not self._eco or not self._eco._ng:
            return None

        try:
            nodes = self._eco._ng.nodes
            if not nodes:
                return None

            total_outcomes = ng_stats.get("total_outcomes", 0)
            if total_outcomes == 0:
                return None

            dead_count = 0
            for node in nodes.values():
                firing_rate = node.activation_count / max(1, total_outcomes)
                if firing_rate < self._config.min_firing_rate:
                    dead_count += 1

            dead_pct = dead_count / len(nodes)
            # Flag if more than 50% of nodes are effectively dead
            if dead_pct > self._config.dead_node_threshold:
                return HealthIssue(
                    category="low_firing_rate",
                    severity=min(1.0, dead_pct),
                    description=(
                        f"{dead_count}/{len(nodes)} nodes below min firing rate "
                        f"({self._config.min_firing_rate}). "
                        f"Substrate carrying {dead_pct:.0%} dead weight."
                    ),
                    metadata={
                        "dead_nodes": dead_count,
                        "total_nodes": len(nodes),
                        "dead_pct": dead_pct,
                    },
                )

        except Exception as exc:
            logger.debug("Firing rate check failed: %s", exc)

        return None

    def _check_novelty_saturation(self) -> Optional[HealthIssue]:
        """Check if substrate is losing discriminative power.

        Novelty saturation occurs when the substrate has seen so many
        patterns that everything registers as "known" — novelty scores
        cluster near 0.0, and the system can't distinguish genuinely
        new failures from variations of known ones.
        """
        if not self._eco or not self._eco._ng:
            return None

        try:
            # Sample a few random embeddings and check novelty
            rng = np.random.RandomState(int(time.time()) % (2**31))
            dim = self._eco._ng.config.get("embedding_dim", 384)
            sample_count = self._config.novelty_probe_count
            low_novelty_count = 0

            for _ in range(sample_count):
                probe = rng.randn(dim).astype(np.float32)
                probe = probe / (np.linalg.norm(probe) + 1e-8)
                novelty = self._eco.detect_novelty(probe)
                if novelty < (1.0 - self._config.novelty_saturation_threshold):
                    low_novelty_count += 1

            saturation = low_novelty_count / sample_count

            if saturation >= self._config.novelty_saturation_threshold:
                return HealthIssue(
                    category="novelty_saturation",
                    severity=saturation,
                    description=(
                        f"Novelty saturation at {saturation:.0%}: "
                        f"random probes score low novelty. "
                        f"Substrate may be losing discriminative power."
                    ),
                    metadata={
                        "saturation": saturation,
                        "sample_count": sample_count,
                        "low_novelty_count": low_novelty_count,
                    },
                )

        except Exception as exc:
            logger.debug("Novelty saturation check failed: %s", exc)

        return None

    # -----------------------------------------------------------------
    # Proactive repair
    # -----------------------------------------------------------------

    def _trigger_proactive_repair(self, issue: HealthIssue) -> None:
        """Trigger a proactive diagnosis for a detected health issue."""
        if self._engine is None:
            return

        try:
            description = f"[HealthMonitor] {issue.description}"
            self._engine.diagnose(
                description=description,
                metadata={
                    "source": "health_monitor",
                    "category": issue.category,
                    "severity": issue.severity,
                    **issue.metadata,
                },
                source="healing_collective",
            )
            logger.info(
                "Proactive repair triggered for %s (severity=%.2f)",
                issue.category, issue.severity,
            )
        except Exception as exc:
            logger.warning("Proactive repair trigger failed: %s", exc)

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return health monitor statistics."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "check_count": self._check_count,
            "issues_detected": self._issues_detected,
            "interval_seconds": self._config.interval_seconds,
            "last_healthy": (
                self._last_report.healthy if self._last_report else None
            ),
        }
