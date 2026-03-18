"""
The Healing Collective — Configuration Module

Loads configuration from ~/.et_modules/healing_collective/config.yaml
with sensible defaults.  The Collective requires zero configuration for
basic operation — all parameters have production-ready defaults.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: HealingCollectiveConfig dataclass with YAML loading and defaults.
#   Why:  PRD §7 specifies all config via config.yaml with absent keys
#         using defaults.  Centralizing config prevents magic numbers.
#   Settings: See DEFAULT_CONFIG below — all values from PRD §7.
#   How:  Dataclass with from_yaml() classmethod.  Nested namespaces
#         (health_monitor.*, congregation.*) use sub-dataclasses.
# -------------------
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("healing_collective.config")


# ---------------------------------------------------------------------------
# Sub-configs for nested namespaces
# ---------------------------------------------------------------------------

@dataclass
class HealthMonitorConfig:
    """Active health monitoring configuration."""
    enabled: bool = True
    interval_seconds: int = 120
    weight_divergence_threshold: float = 2.0
    min_firing_rate: float = 0.001
    novelty_saturation_threshold: float = 0.95


@dataclass
class CongregationConfig:
    """Congregation deliberation configuration."""
    confidence_spread: float = 0.15
    max_candidates: int = 4
    timeout_seconds: float = 10.0
    require_for_host_repairs: bool = True


@dataclass
class CompressionConfig:
    """Pattern compression configuration."""
    trigger_pct: float = 0.80
    similarity_threshold: float = 0.85
    min_cluster: int = 3
    cycle_days: int = 7


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class HealingCollectiveConfig:
    """Full configuration for The Healing Collective.

    All values have PRD-specified defaults.  Override via config.yaml.
    """

    # Confidence thresholds (PRD §2.2.5)
    confidence_auto_execute: float = 0.70
    confidence_recommend: float = 0.40
    confidence_host_premium: float = 0.15

    # Repair safety (PRD §2.2.8)
    repair_cooldown_seconds: int = 300
    revert_window_seconds: int = 60

    # DVS (PRD §2.2.3)
    dvs_max_entries: int = 10000
    dvs_persistence_format: str = "msgpack"
    dvs_search_top_k: int = 10

    # Failure detection (substrate-based, replaces static regex patterns)
    # A message is routed to the diagnosis engine when DVS similarity to
    # known failure signatures exceeds this threshold, OR when substrate
    # novelty exceeds novelty_routing_threshold (unknown-but-suspicious).
    failure_similarity_threshold: float = 0.40
    novelty_routing_threshold: float = 0.85

    # Checkpointing (PRD §6.2)
    checkpoint_interval_seconds: int = 300

    # Nested configs
    health_monitor: HealthMonitorConfig = field(default_factory=HealthMonitorConfig)
    congregation: CongregationConfig = field(default_factory=CongregationConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "HealingCollectiveConfig":
        """Load config from YAML file, merging with defaults.

        Args:
            path: Path to config.yaml.  If None, uses the standard
                  location: ~/.et_modules/healing_collective/config.yaml

        Returns:
            HealingCollectiveConfig with overrides applied.
        """
        if path is None:
            path = os.path.join(
                Path.home(), ".et_modules", "healing_collective", "config.yaml"
            )

        config = cls()

        if not os.path.exists(path):
            logger.debug("No config file at %s, using defaults", path)
            return config

        try:
            import yaml

            with open(path, "r") as f:
                raw = yaml.safe_load(f) or {}

            config._apply_overrides(raw)
            logger.info("Config loaded from %s", path)

        except Exception as exc:
            logger.warning("Failed to load config from %s: %s (using defaults)", path, exc)

        return config

    def _apply_overrides(self, raw: Dict[str, Any]) -> None:
        """Apply raw YAML dict overrides to this config instance."""
        # Top-level scalar fields
        for key in [
            "confidence_auto_execute", "confidence_recommend",
            "confidence_host_premium", "repair_cooldown_seconds",
            "revert_window_seconds", "dvs_max_entries",
            "dvs_persistence_format", "dvs_search_top_k",
            "checkpoint_interval_seconds",
        ]:
            if key in raw:
                setattr(self, key, raw[key])

        # Nested: health_monitor.*
        if "health_monitor" in raw and isinstance(raw["health_monitor"], dict):
            hm = raw["health_monitor"]
            for key in [
                "enabled", "interval_seconds", "weight_divergence_threshold",
                "min_firing_rate", "novelty_saturation_threshold",
            ]:
                if key in hm:
                    setattr(self.health_monitor, key, hm[key])

        # Nested: congregation.*
        if "congregation" in raw and isinstance(raw["congregation"], dict):
            cg = raw["congregation"]
            for key in [
                "confidence_spread", "max_candidates",
                "timeout_seconds", "require_for_host_repairs",
            ]:
                if key in cg:
                    setattr(self.congregation, key, cg[key])

        # Nested: compression.*
        if "compression" in raw and isinstance(raw["compression"], dict):
            cm = raw["compression"]
            for key in [
                "trigger_pct", "similarity_threshold",
                "min_cluster", "cycle_days",
            ]:
                if key in cm:
                    setattr(self.compression, key, cm[key])
