"""
The Healing Collective — Health Monitor

Continuous background monitoring of the NG-Lite substrate health.
Detects anomalies proactively — weight divergence, dead nodes, novelty
saturation — and triggers preventive repairs before failures propagate.

The Health Monitor is the Collective's immune system: always running,
mostly quiet, occasionally nudging the substrate back toward health.

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — #354: reconnect weight-divergence + firing-rate checks
#   What: _check_weight_divergence() and _check_firing_rates() were stubbed `return None`
#         unconditionally since 2026-04-08 (Josh's own commit d877888, "tract-only mode" —
#         self._eco._ng no longer exists once ng_ecosystem=None). Restored via a REAL bucket:
#         a new _bucket_ng_metrics() reads metrics:neurograph:* directly from the Commons
#         (commons.bucket_recent) — HealthMonitor is its own consumer, no data handed to it
#         by the hook. NG deposits RAW weight_mean/weight_std/firing_rate_mean/firing_rate_std
#         (first/second moments of already-native per-element quantities — synapse.weight,
#         node.firing_rate_ema; no threshold applied on NG's side). Both checks restore the
#         EXACT original formula from git history (commit d877888's parent) — divergence =
#         std/mean compared against weight_divergence_threshold, same severity calc.
#         _check_firing_rates adapted: the original computed per-node activation_count/
#         total_outcomes and a dead-node PERCENTAGE — that per-node lifetime data no longer
#         exists in this architecture (Tier-2 per-module ng_lite instances are gone). Uses
#         firing_rate_mean directly against min_firing_rate as a substrate-wide proxy for the
#         same intent ("is the substrate carrying too much dead weight") — an honest
#         approximation, not a byte-identical restoration, documented as such (LAW 3).
#   Why:  First correction attempt (this session) drifted toward two anti-patterns Josh
#         caught: (1) NG pre-computing a "divergence ratio"/"dead percentage" verdict before
#         depositing (LAW 7 — classification belongs at extraction, not at deposit); (2) the
#         hook pre-fetching/EWMA-ing values and pushing them into HealthMonitor via a setter
#         (a push/move-the-data pattern, not a bucket). Corrected: NG deposits raw moments
#         only; HealthMonitor dips its own bucket directly, same as every other THC consumer.
#   How:  _bucket_ng_metrics() — get_commons(), bucket_recent(with_metadata=True), filter
#         metrics:neurograph:*, return the newest deposit's metadata dict (fail-soft, no
#         Commons/no data → {}). Both checks call it independently (fresh read each check
#         cycle — 120s interval, no staleness risk worth caching).
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

        ng_stats = None
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

    def _bucket_ng_metrics(self) -> Dict[str, Any]:
        """Bucket the newest metrics:neurograph:* deposit from the Commons (#354).

        HealthMonitor's OWN direct bucket read — no data handed to it by the hook or any
        other collaborator. Returns the raw metadata dict (weight_mean/weight_std/
        firing_rate_mean/firing_rate_std/...) with no interpretation applied; {} if no
        Commons or no matching deposit yet (fail-soft).
        """
        try:
            from commons import get_commons
            commons = get_commons()
        except Exception:
            return {}
        if commons is None:
            return {}
        try:
            recs = commons.bucket_recent(limit=50, with_metadata=True)
        except Exception as exc:
            logger.debug("HealthMonitor Commons metrics bucket failed: %s", exc)
            return {}
        newest_ts, newest_meta = -1.0, {}
        for target_id, _w, _r, meta in recs:
            if not target_id.startswith("metrics:neurograph:") or not isinstance(meta, dict):
                continue
            ts = meta.get("timestamp", 0.0)
            if ts >= newest_ts:
                newest_ts, newest_meta = ts, meta
        return newest_meta

    def _check_weight_divergence(self, ng_stats: Dict[str, Any]) -> Optional[HealthIssue]:
        """Check if synapse weights are dangerously diverged (#354, restored via Commons bucket).

        Weight divergence indicates the substrate has learned extreme
        associations — some synapses near 1.0, others near 0.0 — which
        reduces its ability to adapt to new patterns.
        """
        meta = self._bucket_ng_metrics()
        mean_weight = meta.get("weight_mean")
        std_dev = meta.get("weight_std")
        if mean_weight is None or std_dev is None:
            return None

        divergence = std_dev / mean_weight if mean_weight > 0 else std_dev

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
                },
            )
        return None

    def _check_firing_rates(self, ng_stats: Dict[str, Any]) -> Optional[HealthIssue]:
        """Check for substrate-wide underactivity (#354, restored via Commons bucket).

        Nodes that never fire are wasting capacity. The original check computed a
        per-node dead-node PERCENTAGE from lifetime activation_count — that per-node
        data no longer exists in this architecture (Tier-2 per-module ng_lite instances
        are gone). This uses firing_rate_mean directly against min_firing_rate as a
        substrate-wide proxy for the same intent: is the substrate carrying too much
        dead weight. Honest approximation, not a byte-identical restoration (LAW 3).
        """
        meta = self._bucket_ng_metrics()
        firing_rate_mean = meta.get("firing_rate_mean")
        if firing_rate_mean is None:
            return None

        if firing_rate_mean < self._config.min_firing_rate:
            return HealthIssue(
                category="low_firing_rate",
                severity=min(1.0, 1.0 - (firing_rate_mean / max(self._config.min_firing_rate, 1e-9))),
                description=(
                    f"Substrate-wide firing rate ({firing_rate_mean:.4f}) below "
                    f"min_firing_rate ({self._config.min_firing_rate}). "
                    f"Substrate may be carrying dead weight."
                ),
                metadata={
                    "firing_rate_mean": firing_rate_mean,
                    "firing_rate_std": meta.get("firing_rate_std"),
                },
            )
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
