# Changelog

All notable changes to The Healing Collective will be documented in this file.

## [0.2.0] - 2026-02-26

### Added
- **Diagnosis Engine** (`core/diagnosis_engine.py`): Seven-step pipeline
  (Observe → Recognize → Recall → Propose → Validate → Execute → Learn)
  with confidence thresholds and cooldown management.
- **NG-Lite-augmented DVS search**: Search routes through substrate's
  learned topology with multi-factor ranking (activation, cosine similarity,
  recency, success rate).
- **Five additional repair primitives**: `config_adjust`, `retry_with_backoff`,
  `ng_lite_rebalance`, `checkpoint_restore`, `connection_pool_reset`.
- **Outcome learning**: Successful repairs strengthen substrate pathways;
  novel failures enter cautious mode with lower confidence ceiling.
- **Checkpoint lifecycle**: Periodic timer, post-repair save, SIGTERM handler.
  Atomic write (temp + rename) for consistency.

## [0.1.0] - 2026-02-26

### Added
- **Module scaffolding**: `et_module.json` (v2 schema), `install.sh`,
  `requirements.txt`, directory structure.
- **Vendored ecosystem files**: `ng_lite.py`, `ng_peer_bridge.py`,
  `ng_ecosystem.py`, `openclaw_adapter.py` (verbatim from TrollGuard).
- **OpenClaw hook** (`healing_collective_hook.py`): `OpenClawAdapter`
  subclass with singleton `get_instance()`, message scanning for failure
  indicators, `report_failure()` host API, `get_healing_status()`.
- **Diagnostic Vector Store** (`core/dvs.py`): msgpack-persisted vector DB
  with LRU eviction, cosine similarity search, JSON debug export.
- **Repair Primitives** (`core/repair_primitives.py`): `RepairPrimitive` ABC
  with `validate()`/`execute()` contract, `ValidationResult`/`ExecutionResult`
  data classes, three Phase 1 primitives (`ProcessRestart`, `CacheClear`,
  `LogAndRecommend`).
- **Configuration** (`core/config.py`): YAML-based config with PRD defaults.
- **Test suite**: 78 tests covering config, DVS, primitives, engine, and hook.
