# Changelog

All notable changes to The Healing Collective will be documented in this file.

## [0.4.0] - 2026-02-27

### Added
- **Pattern Compression** (`core/compression.py`): DVS compaction engine with
  greedy agglomerative clustering, pattern synthesis, and threshold/periodic
  triggering.  Similar entries are replaced with COMPRESSED_PATTERN entries
  that preserve aggregate knowledge.
- **Tier 3 Coordinator** (`core/tier3_upgrade.py`): Cluster-wide repair
  knowledge sharing.  Broadcasts successful repairs to shared learning
  directory; syncs peer repair records into local DVS at startup and
  periodically; provides cluster-aggregated confidence scoring.

### Changed
- **Diagnosis Engine**: Tier 3 broadcast fires after successful repair
  execution, sharing outcomes with the entire cluster.
- **Hook**: Compression timer (hourly check), Tier 3 startup sync,
  updated shutdown handler to stop all background threads.
- Version bump: 0.3.0 → 0.4.0 (Alpha).

## [0.3.0] - 2026-02-27

### Added
- **Health Monitor** (`core/health_monitor.py`): Background substrate health
  monitoring with three checks — weight divergence, firing rate (dead nodes),
  and novelty saturation.  Triggers proactive repairs through the diagnosis
  engine when issues are detected.
- **Congregation** (`core/congregation.py`): Peer deliberation for uncertain
  repairs.  When confidence falls in the recommendation zone, polls peer
  modules' shared learning events for consensus.  Similarity-weighted vote
  aggregation adjusts confidence up or down.

### Changed
- **Diagnosis Engine**: Congregation consulted between Propose and Validate
  when action is "recommend"; consensus can elevate to auto-execute.
  Added `_congregation` and `_tier3` optional collaborators.
- **Hook**: Health Monitor and Congregation initialized after engine,
  wired via post-init setters to break circular dependencies.
  `_module_stats()` now includes Phase 3+4 telemetry.
- Test suite expanded: 78 → 120 tests.

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
