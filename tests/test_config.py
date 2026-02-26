"""
Tests for core/config.py — Configuration loading and defaults.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import os
import tempfile

import pytest
import yaml

from core.config import (
    CongregationConfig,
    CompressionConfig,
    HealingCollectiveConfig,
    HealthMonitorConfig,
)


class TestConfigDefaults:
    """Verify all PRD-specified defaults are correct."""

    def test_confidence_thresholds(self):
        cfg = HealingCollectiveConfig()
        assert cfg.confidence_auto_execute == 0.70
        assert cfg.confidence_recommend == 0.40
        assert cfg.confidence_host_premium == 0.15

    def test_repair_safety(self):
        cfg = HealingCollectiveConfig()
        assert cfg.repair_cooldown_seconds == 300
        assert cfg.revert_window_seconds == 60

    def test_dvs_defaults(self):
        cfg = HealingCollectiveConfig()
        assert cfg.dvs_max_entries == 10000
        assert cfg.dvs_persistence_format == "msgpack"
        assert cfg.dvs_search_top_k == 10

    def test_checkpoint_interval(self):
        cfg = HealingCollectiveConfig()
        assert cfg.checkpoint_interval_seconds == 300

    def test_health_monitor_defaults(self):
        cfg = HealingCollectiveConfig()
        hm = cfg.health_monitor
        assert hm.enabled is True
        assert hm.interval_seconds == 120
        assert hm.weight_divergence_threshold == 2.0
        assert hm.min_firing_rate == 0.001
        assert hm.novelty_saturation_threshold == 0.95

    def test_congregation_defaults(self):
        cfg = HealingCollectiveConfig()
        cg = cfg.congregation
        assert cg.confidence_spread == 0.15
        assert cg.max_candidates == 4
        assert cg.timeout_seconds == 10.0
        assert cg.require_for_host_repairs is True

    def test_compression_defaults(self):
        cfg = HealingCollectiveConfig()
        cm = cfg.compression
        assert cm.trigger_pct == 0.80
        assert cm.similarity_threshold == 0.85
        assert cm.min_cluster == 3
        assert cm.cycle_days == 7


class TestConfigYAML:
    """Test YAML loading and override behavior."""

    def test_missing_file_uses_defaults(self):
        cfg = HealingCollectiveConfig.from_yaml("/nonexistent/path.yaml")
        assert cfg.confidence_auto_execute == 0.70

    def test_override_top_level(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"confidence_auto_execute": 0.80, "dvs_max_entries": 5000}, f)
            f.flush()

            cfg = HealingCollectiveConfig.from_yaml(f.name)
            assert cfg.confidence_auto_execute == 0.80
            assert cfg.dvs_max_entries == 5000
            # Unspecified values remain default
            assert cfg.confidence_recommend == 0.40

        os.unlink(f.name)

    def test_override_nested(self):
        data = {
            "health_monitor": {
                "enabled": False,
                "interval_seconds": 60,
            },
            "congregation": {
                "max_candidates": 8,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()

            cfg = HealingCollectiveConfig.from_yaml(f.name)
            assert cfg.health_monitor.enabled is False
            assert cfg.health_monitor.interval_seconds == 60
            # Unspecified nested value stays default
            assert cfg.health_monitor.weight_divergence_threshold == 2.0
            assert cfg.congregation.max_candidates == 8

        os.unlink(f.name)

    def test_corrupt_yaml_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(":::invalid yaml\n")
            f.flush()

            cfg = HealingCollectiveConfig.from_yaml(f.name)
            assert cfg.confidence_auto_execute == 0.70

        os.unlink(f.name)
