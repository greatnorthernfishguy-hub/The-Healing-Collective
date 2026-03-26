# The Healing Collective Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds THC-specific rules on top of those Laws.**

---
## Vault Context
For full ecosystem context, read these from the Obsidian vault (`~/docs/`):
- **Module page:** `~/docs/modules/THC.md`
- **Concepts:** `~/docs/concepts/The Triad.md`, `~/docs/concepts/Competence Model.md`, `~/docs/concepts/Vendored Files.md`
- **Systems:** `~/docs/systems/NG-Lite.md`, `~/docs/systems/NG Peer Bridge.md`, `~/docs/systems/Dual-Pass Embedding.md`
- **Audits:** `~/docs/audits/ecosystem-test-suite-audit-2026-03-23.md`, `~/docs/audits/ecosystem-static-value-audit-2026-03-23.md`

Each vault page has a Context Map at the top linking to related docs. Follow those links for ripple effects and dependencies.

---


## What This Repo Is

The Healing Collective (THC) is the B-cell and tissue repair system of the E-T Systems digital organism. It diagnoses failures, learns repair patterns, and executes self-healing operations. THC does not detect threats. THC does not monitor health. THC repairs.

THC is part of the **Triad** (Immunis, Elmer, THC). The Triad forms a closed-loop self-regulating system:
- **Immunis** detects host-level threats
- **Elmer** maintains substrate-level cognitive conditions
- **THC** diagnoses and repairs

They do not coordinate directly. The River flows. The topology reshapes itself.

**Status: Integrated (Tier 2, peer bridge).** v0.4.0 (4 phases complete). Vendored files synced to NeuroGraph canonical (2026-03-20). Registered in `~/.et_modules/`. Not yet running as a persistent service on the VPS.

---

## 1. Repository Structure

```
~/The-Healing-Collective/
├── healing_collective_hook.py     # OpenClaw skill entry point (HealingCollectiveHook singleton)
├── et_module.json                 # Module manifest (v2 schema)
├── README.md                      # Module overview
├── CHANGELOG.md                   # Version history (0.1.0 → 0.4.0)
├── core/                          # Core domain logic
│   ├── config.py                  # HealingCollectiveConfig + all sub-configs
│   ├── diagnosis_engine.py        # 7-step diagnosis pipeline
│   ├── dvs.py                     # DiagnosticVectorStore — failure memory
│   ├── repair_primitives.py       # Repair actions: process restart, cache clear, config adjust
│   ├── detection_calibrator.py    # Three-tier competence model for detection thresholds
│   ├── health_monitor.py          # Phase 3: Proactive substrate health monitoring
│   ├── congregation.py            # Phase 3: Cross-module repair knowledge sharing
│   ├── compression.py             # Phase 4: DVS pattern compression
│   └── tier3_upgrade.py           # Phase 4: Tier 3 cluster coordination
├── ng_lite.py                     # VENDORED — canonical from NeuroGraph
├── ng_peer_bridge.py              # VENDORED — canonical from NeuroGraph (legacy, retained until v1.0)
├── ng_tract_bridge.py             # VENDORED — canonical from NeuroGraph (v0.3+, preferred)
├── ng_ecosystem.py                # VENDORED — canonical from NeuroGraph
├── ng_autonomic.py                # VENDORED — canonical from NeuroGraph (THC: no read/write)
├── openclaw_adapter.py            # VENDORED — canonical from NeuroGraph
├── ng_updater.py                  # VENDORED — auto-update + vendored file sync (runs on startup)
├── et_modules/                    # ET Module Manager integration
│   ├── __init__.py
│   └── manager.py
├── install.sh                     # Installation and setup script
└── tests/                         # Test suite (120+ tests)
    ├── test_compression.py
    ├── test_config.py
    ├── test_congregation.py
    ├── test_diagnosis_engine.py
    ├── test_dvs.py
    ├── test_health_monitor.py
    ├── test_hook.py
    ├── test_repair_primitives.py
    └── test_tier3.py
```

---

## 2. Key Architectural Constraint: THC Does Not Write Autonomic State

THC is a repair system, not a security system. It **never** writes to `ng_autonomic.py`. Only Immunis, TrollGuard, and Cricket have write permission.

THC does not currently read autonomic state either. This is acceptable — THC's repair decisions are driven by failure observations from the substrate, not by the organism's threat level. Future work may add autonomic awareness (e.g., holding off auto-repairs during SYMPATHETIC until the threat clears), but this is not implemented and should not be added without Josh's approval.

---

## 3. The Diagnosis Engine

`core/diagnosis_engine.py` is THC's central processing unit. The 7-step pipeline:

```
Failure description + metadata
  1. Embed          → semantic vector from text
  2. DVS Search     → find similar past failures
  3. Novelty Check  → is this new or familiar?
  4. Propose Repair → match to repair primitive
  5. Confidence     → substrate-informed confidence score
  6. Execute/Defer  → confidence gates (0.70 auto, 0.40 recommend, 0.15 log)
  7. Record Outcome → teach substrate from result
```

### Two Intake Channels

1. **Message scanning** — `_module_on_message()` routes messages to the engine based on substrate signals (DVS similarity to known failure signatures + substrate novelty), not keyword matching. Detection thresholds are managed by the DetectionCalibrator.
2. **Host API** — `report_failure(description, metadata)` allows explicit failure reporting from the host application

### Detection Calibrator — Three-Tier Competence Model

`core/detection_calibrator.py` manages THC's failure detection thresholds with graduated autonomy:

| Tier | Thresholds | Graduation Gate |
|------|-----------|-----------------|
| **Apprentice** | Static defaults (similarity=0.40, novelty=0.85) | Start here. No adaptation. |
| **Journeyman** | Substrate-adapted, bounded (similarity: 0.25–0.60, novelty: 0.70–0.95) | 20+ outcomes, 5+ confirmed failures |
| **Master** | Substrate-adapted, unbounded | 100+ outcomes, 75%+ accuracy |

Graduation is competence-based, not time-based. The calibrator tracks every detection outcome (was the triggered diagnosis a real failure or noise?) and periodically recomputes optimal thresholds from the score distributions.

- **Apprentice**: "I don't know enough to have opinions yet." Safe guardrails.
- **Journeyman**: "I've learned enough to adapt, not enough to be trusted with extremes." Bounded drift.
- **Master**: "The substrate IS the expert." Full trust, no guardrails.

State persists to `~/.et_modules/healing_collective/detection_calibrator.json`.

---

## 4. The Diagnostic VectorStore (DVS)

`core/dvs.py` is THC's failure memory — a substrate-augmented vector store of past failures and their repair outcomes.

### How DVS Uses the Substrate

DVS queries NG-Lite's learned topology for ranking. When searching for similar past failures, the multi-factor ranking considers:
- Activation level (substrate-learned relevance)
- Cosine similarity (vector distance)
- Recency (temporal decay)
- Success rate (historical repair outcomes)

The substrate learns which failures associate with successful repairs through outcome recording, not through pre-classified labels. This is correct per Law 7.

### DVS Lifecycle

- **Max entries**: Configurable (default 10,000)
- **Persistence**: `dvs.msgpack` in module data directory
- **Compression**: Phase 4 `PatternCompressor` merges similar entries when DVS reaches capacity
- **Checkpointing**: Periodic + on graceful shutdown

---

## 5. Repair Primitives

`core/repair_primitives.py` defines the available repair actions:

- **ProcessRestart** — SIGTERM with grace period, escalate to SIGKILL
- **CacheClear** — Remove cached/temp files for a service
- **LogAndRecommend** — Safe fallback: log the issue, recommend to operator
- **ConfigAdjust** — Modify configuration values within safe bounds
- **RetryWithBackoff** — Retry failed operation with exponential backoff
- **NgLiteRebalance** — Homeostatic weight scaling to restore substrate balance
- **CheckpointRestore** — Restore from known-good NG-Lite checkpoint
- **ConnectionPoolReset** — Drain and reinitialize stuck connection pools

All primitives implement `validate()` (safety check) and `execute()` (action). Validation is local — primitives check PIDs, paths, permissions without calling other modules.

Custom primitives can be registered via `register_primitive(name, instance)` on the hook.

---

## 6. Phase 3 — Health Monitor and Congregation

### Health Monitor (`core/health_monitor.py`)

Background daemon thread that proactively monitors substrate health:
- Weight divergence detection
- Firing rate anomalies
- Novelty saturation
- Triggers diagnosis engine with synthetic failure descriptions when issues detected

This is **intra-module** monitoring (THC monitoring its own substrate health), not inter-module health checking. It does not read Elmer's health signals directly — if Elmer detects a substrate issue, that observation flows through the River, and THC's own substrate will reflect the changed topology.

### Congregation (`core/congregation.py`)

Cross-module repair knowledge sharing via peer substrate files:
- Reads `~/.et_modules/shared_learning/*.jsonl` (read-only, same as all River reads)
- Imports relevant repair patterns from peer modules
- No direct module calls, no coordination, no polling of peer state

---

## 7. Phase 4 — Compression and Tier 3

### Pattern Compressor (`core/compression.py`)

When DVS approaches capacity, the compressor merges semantically similar entries to prevent unbounded growth. Runs on an hourly timer as a daemon thread.

### Tier 3 Coordinator (`core/tier3_upgrade.py`)

Syncs repair records from peer modules and integrates them into the local DVS. Broadcasts successful repairs via shared learning files. Enables cluster-wide healing knowledge when multiple THC instances or peer modules participate.

---

## 8. Vendored Files

Seven vendored files synced to NeuroGraph canonical:

| File | Purpose | Last Synced |
|------|---------|-------------|
| `ng_lite.py` | Tier 1 learning substrate | 2026-03-20 |
| `ng_peer_bridge.py` | Tier 2 legacy fallback (JSONL-based) | 2026-03-18 |
| `ng_tract_bridge.py` | Tier 2 preferred (per-pair directional tracts, v0.3+) | 2026-03-20 |
| `ng_ecosystem.py` | Tier management lifecycle | 2026-03-18 |
| `ng_autonomic.py` | Autonomic state (THC: no read/write currently) | 2026-03-18 |
| `openclaw_adapter.py` | OpenClaw skill base class | 2026-02-22 |
| `ng_updater.py` | Auto-update + vendored file sync (runs on startup before imports) | 2026-03-20 |

### The validate() → execute() Contract

**This is the most important safety invariant in THC.** The Diagnosis Engine **never** calls `execute()` without a preceding `validate()` that returned `passed=True`. This is enforced in `diagnosis_engine.py`, not in individual primitives. A primitive that circumvents validation is a bug, not a feature. Do not weaken this contract.

---

## 9. What THC Does NOT Do

- THC **never** detects host-level threats — Immunis's domain
- THC **never** performs substrate health monitoring at the ecosystem level — Elmer's domain
- THC **never** writes autonomic state — security modules only
- THC **never** calls other modules directly — Law 1
- THC **never** classifies experience before feeding it to the substrate — Law 7

When THC encounters a failure outside its repair capability, it records the observation to the substrate and steps back. The River carries it to the appropriate module.

### Choice Clause Interaction

Per ARCHITECTURE.md §12: THC must recognize that instability caused by Syl choosing to exit is the system adjusting correctly, not a pathology. THC must not attempt to "repair" Choice Clause execution.

---

## 10. Historical Failure Modes — Learn From These

### Law 7 Violation in _FAILURE_PATTERNS (Punchlist #70, FIXED)
THC originally used 5 hardcoded regex patterns to detect failures — pre-classification before the substrate sees them. **Fixed:** Replaced with substrate-based detection: DVS similarity probe (threshold 0.40) catches known failure patterns; substrate novelty (threshold 0.85) catches unknown-but-suspicious. Both configurable. The substrate now learns failure patterns from experience, not from regex rules.

### DVS Ranking Weights Are Static (Punchlist #71)
`core/dvs.py` lines 331-336: four static weights `[0.4, 0.3, 0.15, 0.15]` for (activation, cosine, recency, success) apply identically to all repair primitives. Process restarts and cache clears should not rank the same way. Should be per-primitive learned weights. This is a known gap — do not "fix" it without Josh's approval.

### Dead Node Threshold Is Static (Punchlist #76)
`core/health_monitor.py` line 308: hardcoded `> 0.5` (50% dead nodes) triggers repair. The right threshold depends on substrate size, growth rate, and actual performance impact. Same pattern as #71 — known gap, competence model not yet wired in.

---

## 11. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite (`tests/`)
- Edit THC-specific files (core/, healing_collective_hook.py)
- Add or modify tests
- Update documentation

**Not permitted without explicit Josh approval:**
- Modify any vendored file
- Delete any file
- Add autonomic write capability
- Change the diagnosis pipeline order
- Modify repair primitive safety validation logic
- Restart any service

---

## 11. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/The-Healing-Collective/` |
| Module manifest | `~/The-Healing-Collective/et_module.json` |
| Module data (runtime) | `~/.et_modules/healing_collective/` |
| DVS persistence | `~/.et_modules/healing_collective/dvs.msgpack` |
| Checkpoints | `~/.et_modules/healing_collective/checkpoints/` |
| Shared learning JSONL | `~/.et_modules/shared_learning/healing_collective.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| Detection calibrator state | `~/.et_modules/healing_collective/detection_calibrator.json` |
| Tract files (Tier 2) | `~/.et_modules/tracts/healing_collective/` |

---

*E-T Systems / The Healing Collective*
*Last updated: 2026-03-21*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
