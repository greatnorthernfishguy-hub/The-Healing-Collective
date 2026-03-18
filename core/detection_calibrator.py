"""
Detection Calibrator — Adaptive Failure Detection Thresholds

Manages the three-tier competence model for THC's failure detection:

  APPRENTICE  — Static thresholds from config. No drift. The substrate
                has not seen enough to have opinions yet.
  JOURNEYMAN  — Substrate-informed drift within bounded range. Enough
                experience to adapt, not enough to be trusted with extremes.
  MASTER      — Unbounded substrate authority. Thresholds go wherever the
                learned distribution says they should. The substrate IS
                the expert.

Graduation is competence-based, not time-based. More experience with
better outcomes = more autonomy. The gates measure what the system has
learned, not how long it has been running.

The calibrator observes every detection outcome (was the triggered
diagnosis a real failure or noise?) and periodically recomputes
optimal thresholds from the distribution of scores.

# ---- Changelog ----
# [2026-03-18] Claude (CC) — Initial creation
# What: Three-tier competence model for failure detection thresholds.
# Why: Static thresholds are bootstrap scaffolding. The substrate should
#   learn what similarity/novelty scores actually predict real failures,
#   with autonomy proportional to demonstrated competence.
# How: Track (score, was_real_failure) observations. Compute optimal
#   thresholds from score distributions. Clamp to bounds in Journeyman
#   mode. No clamp in Master mode. Graduate based on outcome count +
#   accuracy metrics.
# -------------------
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("healing_collective.calibrator")


class CompetenceTier(str, enum.Enum):
    APPRENTICE = "apprentice"
    JOURNEYMAN = "journeyman"
    MASTER = "master"


@dataclass
class CalibrationConfig:
    """Configuration for the detection calibrator.

    All values are fair starting values — Elmer-tunable when Elmer
    integrates (Key Decision #7).
    """

    # Apprentice defaults (static thresholds)
    default_similarity_threshold: float = 0.40
    default_novelty_threshold: float = 0.85

    # Journeyman bounds — drift permitted within this range
    similarity_bounds: Tuple[float, float] = (0.25, 0.60)
    novelty_bounds: Tuple[float, float] = (0.70, 0.95)

    # Graduation gates: Apprentice → Journeyman
    journeyman_min_outcomes: int = 20       # Diagnosed failures with feedback
    journeyman_min_true_positives: int = 5  # Confirmed real failures

    # Graduation gates: Journeyman → Master
    master_min_outcomes: int = 100
    master_min_accuracy: float = 0.75  # True positive rate over all outcomes

    # Recompute interval — recalibrate after this many new observations
    recalibrate_every: int = 10


@dataclass
class DetectionObservation:
    """A single observation for calibration learning."""
    similarity_score: float     # DVS similarity when detection triggered
    novelty_score: float        # Substrate novelty when detection triggered
    trigger: str                # "dvs_similarity" or "novelty"
    was_real_failure: bool      # Did diagnosis confirm a real failure?
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DetectionCalibrator:
    """Adaptive threshold calibration with three-tier competence model.

    Usage:
        calibrator = DetectionCalibrator(config, persistence_path)

        # Get current thresholds (tier-appropriate)
        sim_thresh, nov_thresh = calibrator.get_thresholds()

        # After a detection triggers diagnosis, record the outcome
        calibrator.record_outcome(
            similarity_score=0.45,
            novelty_score=0.30,
            trigger="dvs_similarity",
            was_real_failure=True,
        )

        # Check current tier
        tier = calibrator.tier  # APPRENTICE, JOURNEYMAN, or MASTER
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._config = config or CalibrationConfig()
        self._persistence_path = persistence_path

        self._observations: List[DetectionObservation] = []
        self._tier = CompetenceTier.APPRENTICE

        # Current adaptive thresholds (start at defaults)
        self._similarity_threshold = self._config.default_similarity_threshold
        self._novelty_threshold = self._config.default_novelty_threshold

        # Counter for triggering recalibration
        self._observations_since_calibration = 0

        # Load persisted state if available
        if persistence_path:
            self._load(persistence_path)

    @property
    def tier(self) -> CompetenceTier:
        return self._tier

    def get_thresholds(self) -> Tuple[float, float]:
        """Return current (similarity_threshold, novelty_threshold).

        In APPRENTICE mode, returns static defaults.
        In JOURNEYMAN mode, returns substrate-adapted values clamped to bounds.
        In MASTER mode, returns substrate-adapted values without bounds.
        """
        return self._similarity_threshold, self._novelty_threshold

    def record_outcome(
        self,
        similarity_score: float,
        novelty_score: float,
        trigger: str,
        was_real_failure: bool,
    ) -> None:
        """Record a detection outcome for calibration learning.

        Call this after a triggered detection has been diagnosed and
        the outcome is known (real failure vs noise/false positive).
        """
        obs = DetectionObservation(
            similarity_score=similarity_score,
            novelty_score=novelty_score,
            trigger=trigger,
            was_real_failure=was_real_failure,
        )
        self._observations.append(obs)
        self._observations_since_calibration += 1

        # Check graduation
        self._evaluate_graduation()

        # Recalibrate periodically
        if self._observations_since_calibration >= self._config.recalibrate_every:
            self._recalibrate()
            self._observations_since_calibration = 0

        # Persist
        if self._persistence_path:
            self._save(self._persistence_path)

    def stats(self) -> Dict[str, Any]:
        """Calibrator statistics for telemetry."""
        total = len(self._observations)
        true_positives = sum(1 for o in self._observations if o.was_real_failure)
        false_positives = total - true_positives

        return {
            "tier": self._tier.value,
            "total_observations": total,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "accuracy": true_positives / total if total > 0 else 0.0,
            "similarity_threshold": round(self._similarity_threshold, 4),
            "novelty_threshold": round(self._novelty_threshold, 4),
            "default_similarity": self._config.default_similarity_threshold,
            "default_novelty": self._config.default_novelty_threshold,
        }

    # -----------------------------------------------------------------
    # Graduation
    # -----------------------------------------------------------------

    def _evaluate_graduation(self) -> None:
        """Check whether competence warrants tier promotion."""
        total = len(self._observations)
        true_positives = sum(1 for o in self._observations if o.was_real_failure)
        accuracy = true_positives / total if total > 0 else 0.0

        if self._tier == CompetenceTier.APPRENTICE:
            if (
                total >= self._config.journeyman_min_outcomes
                and true_positives >= self._config.journeyman_min_true_positives
            ):
                self._tier = CompetenceTier.JOURNEYMAN
                self._recalibrate()
                logger.info(
                    "Detection calibrator graduated to JOURNEYMAN "
                    "(outcomes=%d, true_positives=%d)",
                    total, true_positives,
                )

        elif self._tier == CompetenceTier.JOURNEYMAN:
            if (
                total >= self._config.master_min_outcomes
                and accuracy >= self._config.master_min_accuracy
            ):
                self._tier = CompetenceTier.MASTER
                self._recalibrate()
                logger.info(
                    "Detection calibrator graduated to MASTER "
                    "(outcomes=%d, accuracy=%.2f)",
                    total, accuracy,
                )

    # -----------------------------------------------------------------
    # Recalibration
    # -----------------------------------------------------------------

    def _recalibrate(self) -> None:
        """Recompute thresholds from observed score distributions.

        APPRENTICE: No-op. Static thresholds.
        JOURNEYMAN: Adapt within bounds.
        MASTER: Adapt without bounds.
        """
        if self._tier == CompetenceTier.APPRENTICE:
            return

        # Separate observations by trigger type
        sim_obs = [o for o in self._observations if o.trigger == "dvs_similarity"]
        nov_obs = [o for o in self._observations if o.trigger == "novelty"]

        # Compute optimal similarity threshold
        if len(sim_obs) >= 5:
            self._similarity_threshold = self._compute_optimal_threshold(
                scores=[(o.similarity_score, o.was_real_failure) for o in sim_obs],
                default=self._config.default_similarity_threshold,
            )

        # Compute optimal novelty threshold
        if len(nov_obs) >= 5:
            self._novelty_threshold = self._compute_optimal_threshold(
                scores=[(o.novelty_score, o.was_real_failure) for o in nov_obs],
                default=self._config.default_novelty_threshold,
            )

        # Apply bounds in Journeyman mode
        if self._tier == CompetenceTier.JOURNEYMAN:
            sim_lo, sim_hi = self._config.similarity_bounds
            nov_lo, nov_hi = self._config.novelty_bounds
            self._similarity_threshold = max(sim_lo, min(sim_hi, self._similarity_threshold))
            self._novelty_threshold = max(nov_lo, min(nov_hi, self._novelty_threshold))

        # Master mode: no clamping. The substrate is the expert.

        logger.info(
            "Recalibrated thresholds (tier=%s): similarity=%.3f, novelty=%.3f",
            self._tier.value,
            self._similarity_threshold,
            self._novelty_threshold,
        )

    @staticmethod
    def _compute_optimal_threshold(
        scores: List[Tuple[float, bool]],
        default: float,
    ) -> float:
        """Find the threshold that best separates real failures from noise.

        Uses a simple approach: find the score that maximizes the gap
        between the mean score of true positives and the mean score of
        false positives. The optimal threshold sits between the two means.

        If there's insufficient separation, returns the default.
        """
        true_scores = [s for s, is_real in scores if is_real]
        false_scores = [s for s, is_real in scores if not is_real]

        if not true_scores or not false_scores:
            # Can't separate without both classes
            if true_scores:
                # Only real failures seen — threshold just below the minimum
                return max(0.0, min(true_scores) - 0.05)
            return default

        true_mean = sum(true_scores) / len(true_scores)
        false_mean = sum(false_scores) / len(false_scores)

        # Midpoint between the two distributions
        midpoint = (true_mean + false_mean) / 2.0

        # If the distributions overlap heavily (means within 0.05),
        # the signal isn't clean enough — lean toward the default
        if abs(true_mean - false_mean) < 0.05:
            return default

        return midpoint

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _save(self, path: str) -> None:
        """Persist calibration state."""
        try:
            state = {
                "tier": self._tier.value,
                "similarity_threshold": self._similarity_threshold,
                "novelty_threshold": self._novelty_threshold,
                "observations": [
                    {
                        "similarity_score": o.similarity_score,
                        "novelty_score": o.novelty_score,
                        "trigger": o.trigger,
                        "was_real_failure": o.was_real_failure,
                        "timestamp": o.timestamp,
                    }
                    for o in self._observations
                ],
            }
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(p)
        except Exception as exc:
            logger.debug("Calibration save failed: %s", exc)

    def _load(self, path: str) -> None:
        """Load persisted calibration state."""
        try:
            p = Path(path)
            if not p.exists():
                return
            state = json.loads(p.read_text())

            self._tier = CompetenceTier(state.get("tier", "apprentice"))
            self._similarity_threshold = state.get(
                "similarity_threshold",
                self._config.default_similarity_threshold,
            )
            self._novelty_threshold = state.get(
                "novelty_threshold",
                self._config.default_novelty_threshold,
            )

            for obs_dict in state.get("observations", []):
                self._observations.append(DetectionObservation(
                    similarity_score=obs_dict["similarity_score"],
                    novelty_score=obs_dict["novelty_score"],
                    trigger=obs_dict["trigger"],
                    was_real_failure=obs_dict["was_real_failure"],
                    timestamp=obs_dict.get("timestamp", 0.0),
                ))

            logger.info(
                "Loaded calibration state: tier=%s, %d observations, "
                "sim=%.3f, nov=%.3f",
                self._tier.value,
                len(self._observations),
                self._similarity_threshold,
                self._novelty_threshold,
            )
        except Exception as exc:
            logger.debug("Calibration load failed: %s", exc)
