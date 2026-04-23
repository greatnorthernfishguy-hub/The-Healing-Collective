"""
Detection Calibrator — Continuous Competence-Based Threshold Adaptation

Manages THC's failure detection thresholds using a continuous per-parameter
competence model. Matches Elmer's TuningSocket pattern.

Competence is a continuous [0, 1] value per threshold parameter ("similarity"
and "novelty"). It grows with accurate detection outcomes (gain=0.05) and
shrinks with regressions (loss=0.10, asymmetric — trust is hard to build, easy
to lose). Bootstrap bounds expand as competence rises — the substrate earns a
wider operating envelope through demonstrated accuracy.

Tier names are descriptive regions only, for telemetry and logging:
  Apprentice region  — competence near 0.0: bootstrap defaults dominate
  Journeyman region  — competence 0.25–0.75: substrate drift begins
  Master region      — competence near 1.0: substrate IS the expert

# ---- Changelog ----
# [2026-04-23] Claude (Sonnet 4.6) — Punchlist #170: migrate to continuous competence
#   What: Replace discrete CompetenceTier enum + graduation gates with continuous
#     per-parameter competence drift matching Elmer's TuningSocket.
#   Why:  Josh directive — tier gates were a mistake. Continuous confidence-weighted
#     drift is the ecosystem standard. Reference: Elmer/core/tuning.py.
#   How:  _competence dict per threshold key ("similarity", "novelty"). Asymmetric
#     gain=0.05/loss=0.10. Bootstrap bounds expand by _BOUNDS_EXPANSION_FACTOR at
#     full competence. No tier state machine, no graduation check. Backwards-compat
#     load: old "tier" field ignored; observations preserved; competence starts at 0.
# [2026-03-18] Claude (CC) — Initial creation
#   What: Three-tier competence model for failure detection thresholds.
#   Why:  Static thresholds are bootstrap scaffolding. The substrate should learn
#     what similarity/novelty scores actually predict real failures, with autonomy
#     proportional to demonstrated competence.
#   How:  Track (score, was_real_failure) observations. Compute optimal thresholds
#     from score distributions. Graduate based on outcome count + accuracy metrics.
# -------------------
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("healing_collective.calibrator")

# Competence update rates. Asymmetric: trust is hard to earn, easy to lose.
_COMPETENCE_GAIN = 0.05    # per accurate detection (true positive)
_COMPETENCE_LOSS = 0.10    # per regression (false positive) — 2× gain

# Minimum outcomes before competence can exceed 0.5.
# Prevents early lucky outcomes from producing overconfidence.
_MIN_OUTCOMES_FOR_CONFIDENCE = 5

# How much bootstrap bounds expand at full competence (c=1.0).
# E.g., similarity_bounds (0.25, 0.60) width=0.35 → expands ±0.175 at c=1:
#   effective bounds (0.075, 0.775)
_BOUNDS_EXPANSION_FACTOR = 0.5

# Minimum trigger-type observations before adapting that threshold.
_MIN_OBS_TO_ADAPT = 5


@dataclass
class CalibrationConfig:
    """Configuration for the detection calibrator."""

    # Bootstrap defaults (static starting points)
    default_similarity_threshold: float = 0.40
    default_novelty_threshold: float = 0.85

    # Bootstrap bounds — expanded by competence at runtime
    similarity_bounds: Tuple[float, float] = (0.25, 0.60)
    novelty_bounds: Tuple[float, float] = (0.70, 0.95)

    # Recalibrate after this many new observations
    recalibrate_every: int = 10


@dataclass
class DetectionObservation:
    """A single observation for calibration learning."""
    similarity_score: float
    novelty_score: float
    trigger: str                # "dvs_similarity" or "novelty"
    was_real_failure: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class DetectionCalibrator:
    """Adaptive threshold calibration with continuous per-parameter competence.

    Competence is a continuous [0, 1] value per threshold parameter. It grows
    with accurate detections and shrinks with false positives. Bootstrap bounds
    expand as competence rises — no discrete tier gates.

    Usage:
        calibrator = DetectionCalibrator(config, persistence_path)
        sim_thresh, nov_thresh = calibrator.get_thresholds()
        calibrator.record_outcome(
            similarity_score=0.45, novelty_score=0.30,
            trigger="dvs_similarity", was_real_failure=True,
        )
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._config = config or CalibrationConfig()
        self._persistence_path = persistence_path

        self._observations: List[DetectionObservation] = []
        self._similarity_threshold = self._config.default_similarity_threshold
        self._novelty_threshold = self._config.default_novelty_threshold

        # Per-parameter competence: continuous [0, 1], starts at 0 (no evidence).
        self._competence: Dict[str, float] = {"similarity": 0.0, "novelty": 0.0}
        self._outcome_counts: Dict[str, int] = {"similarity": 0, "novelty": 0}

        self._observations_since_calibration = 0

        if persistence_path:
            self._load(persistence_path)

    # -------------------------------------------------------------------
    # Competence
    # -------------------------------------------------------------------

    def get_competence(self, key: str) -> float:
        """Return current competence for a parameter. 0 = no evidence."""
        raw = self._competence.get(key, 0.0)
        outcomes = self._outcome_counts.get(key, 0)
        if outcomes < _MIN_OUTCOMES_FOR_CONFIDENCE:
            cap = outcomes / _MIN_OUTCOMES_FOR_CONFIDENCE * 0.5
            return min(raw, cap)
        return raw

    def _update_competence(self, key: str, was_real_failure: bool) -> float:
        """Update competence based on detection accuracy.

        True positive (real failure detected) → gain competence.
        False positive (noise triggered) → lose competence.
        """
        current = self._competence.get(key, 0.0)
        self._outcome_counts[key] = self._outcome_counts.get(key, 0) + 1

        if was_real_failure:
            new = current + _COMPETENCE_GAIN * (1.0 - current)
        else:
            new = current - _COMPETENCE_LOSS * current

        new = max(0.0, min(1.0, new))
        self._competence[key] = new
        logger.debug("Competence %s: %.3f → %.3f", key, current, new)
        return new

    def _effective_bounds(self, key: str) -> Tuple[float, float]:
        """Bootstrap bounds expanded by competence.

        At c=0 returns bootstrap bounds. At c=1 expands by
        _BOUNDS_EXPANSION_FACTOR in each direction.
        """
        c = self.get_competence(key)
        lo, hi = (
            self._config.similarity_bounds
            if key == "similarity"
            else self._config.novelty_bounds
        )
        width = hi - lo
        expansion = width * _BOUNDS_EXPANSION_FACTOR * c
        return (max(0.0, lo - expansion), min(1.0, hi + expansion))

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    @property
    def tier(self) -> str:
        """Descriptive competence region — for telemetry only, not a gate."""
        avg_c = sum(self._competence.values()) / max(len(self._competence), 1)
        if avg_c < 0.25:
            return "apprentice"
        elif avg_c < 0.75:
            return "journeyman"
        return "master"

    def get_thresholds(self) -> Tuple[float, float]:
        """Return (similarity_threshold, novelty_threshold)."""
        return self._similarity_threshold, self._novelty_threshold

    def record_outcome(
        self,
        similarity_score: float,
        novelty_score: float,
        trigger: str,
        was_real_failure: bool,
    ) -> None:
        """Record a detection outcome and update competence."""
        obs = DetectionObservation(
            similarity_score=similarity_score,
            novelty_score=novelty_score,
            trigger=trigger,
            was_real_failure=was_real_failure,
        )
        self._observations.append(obs)
        self._observations_since_calibration += 1

        key = "similarity" if trigger == "dvs_similarity" else "novelty"
        self._update_competence(key, was_real_failure)

        if self._observations_since_calibration >= self._config.recalibrate_every:
            self._recalibrate()
            self._observations_since_calibration = 0

        if self._persistence_path:
            self._save(self._persistence_path)

    def stats(self) -> Dict[str, Any]:
        """Calibrator statistics for telemetry."""
        total = len(self._observations)
        true_positives = sum(1 for o in self._observations if o.was_real_failure)
        return {
            "tier": self.tier,
            "competence_similarity": round(self.get_competence("similarity"), 4),
            "competence_novelty": round(self.get_competence("novelty"), 4),
            "total_observations": total,
            "true_positives": true_positives,
            "false_positives": total - true_positives,
            "accuracy": true_positives / total if total > 0 else 0.0,
            "similarity_threshold": round(self._similarity_threshold, 4),
            "novelty_threshold": round(self._novelty_threshold, 4),
            "default_similarity": self._config.default_similarity_threshold,
            "default_novelty": self._config.default_novelty_threshold,
        }

    # -------------------------------------------------------------------
    # Recalibration
    # -------------------------------------------------------------------

    def _recalibrate(self) -> None:
        """Recompute thresholds from observed score distributions.

        Always runs (no APPRENTICE no-op). Stays at defaults when insufficient
        observations for a trigger type. Bounds expand with competence.
        """
        sim_obs = [o for o in self._observations if o.trigger == "dvs_similarity"]
        nov_obs = [o for o in self._observations if o.trigger == "novelty"]

        if len(sim_obs) >= _MIN_OBS_TO_ADAPT:
            optimal = self._compute_optimal_threshold(
                scores=[(o.similarity_score, o.was_real_failure) for o in sim_obs],
                default=self._config.default_similarity_threshold,
            )
            lo, hi = self._effective_bounds("similarity")
            self._similarity_threshold = max(lo, min(hi, optimal))

        if len(nov_obs) >= _MIN_OBS_TO_ADAPT:
            optimal = self._compute_optimal_threshold(
                scores=[(o.novelty_score, o.was_real_failure) for o in nov_obs],
                default=self._config.default_novelty_threshold,
            )
            lo, hi = self._effective_bounds("novelty")
            self._novelty_threshold = max(lo, min(hi, optimal))

        logger.info(
            "Recalibrated (sim_c=%.3f, nov_c=%.3f): sim=%.3f, nov=%.3f",
            self.get_competence("similarity"),
            self.get_competence("novelty"),
            self._similarity_threshold,
            self._novelty_threshold,
        )

    @staticmethod
    def _compute_optimal_threshold(
        scores: List[Tuple[float, bool]],
        default: float,
    ) -> float:
        """Midpoint between true-positive and false-positive score distributions.

        Returns default if distributions are too similar (gap < 0.05).
        """
        true_scores = [s for s, is_real in scores if is_real]
        false_scores = [s for s, is_real in scores if not is_real]

        if not true_scores or not false_scores:
            if true_scores:
                return max(0.0, min(true_scores) - 0.05)
            return default

        true_mean = sum(true_scores) / len(true_scores)
        false_mean = sum(false_scores) / len(false_scores)

        if abs(true_mean - false_mean) < 0.05:
            return default

        return (true_mean + false_mean) / 2.0

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _save(self, path: str) -> None:
        """Persist calibration state."""
        try:
            state = {
                "competence": self._competence,
                "outcome_counts": self._outcome_counts,
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
        """Load persisted calibration state.

        Backwards compatible: old \"tier\" field silently ignored.
        Competence starts at 0.0 for migrated instances — conservative default.
        """
        try:
            p = Path(path)
            if not p.exists():
                return
            state = json.loads(p.read_text())

            if "competence" in state:
                self._competence.update(state["competence"])
            # Old "tier" field silently dropped — competence starts at 0.0.

            if "outcome_counts" in state:
                self._outcome_counts.update(state["outcome_counts"])

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
                "Loaded calibration state: sim_c=%.3f, nov_c=%.3f, "
                "%d observations, sim=%.3f, nov=%.3f",
                self.get_competence("similarity"),
                self.get_competence("novelty"),
                len(self._observations),
                self._similarity_threshold,
                self._novelty_threshold,
            )
        except Exception as exc:
            logger.debug("Calibration load failed: %s", exc)
