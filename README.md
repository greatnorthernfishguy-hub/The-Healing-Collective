# The Healing Collective

**Adaptive Self-Healing Intelligence for the E-T Systems Ecosystem**

The Healing Collective observes failures across host systems and connected peer modules, learns what repairs work through NG-Lite Hebbian learning, and applies that knowledge automatically. It is NOT a runbook engine — it ships with repair primitives and a learning substrate; the topology of what-fixes-what is learned through experience.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  OpenClaw Skill Interface            │
│        on_message() · recall() · stats()            │
├─────────────────────────────────────────────────────┤
│           healing_collective_hook.py                 │
│    ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│    │  Diagnosis │  │    DVS    │  │   Repair     │  │
│    │  Engine    │  │  (Vector  │  │  Primitives  │  │
│    │  (7-step)  │  │   Store)  │  │  (8 built-in)│  │
│    └───────────┘  └───────────┘  └──────────────┘  │
│    ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│    │  Health   │  │ Congre-   │  │  Pattern     │  │
│    │  Monitor  │  │  gation   │  │  Compression │  │
│    └───────────┘  └───────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────┤
│          NG-Lite Substrate + NGEcosystem             │
│  Tier 1: Standalone → Tier 2: Peer → Tier 3: Cluster│
└─────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
./install.sh

# Or install dependencies only
./install.sh --deps-only

# Check status
./install.sh --status
```

## Usage

### As an OpenClaw Skill

The module is automatically loaded by OpenClaw when configured in SKILL.md:

```yaml
name: healing_collective
autoload: true
hook: healing_collective_hook.py::get_instance
```

### Programmatic API

```python
from healing_collective_hook import get_instance

hc = get_instance()

# Report a failure (Channel 2: Host API)
tracking_id = hc.report_failure(
    "Database connection timeout after 30s",
    metadata={"host": "db.example.com", "port": 5432}
)

# Check status
status = hc.get_healing_status(tracking_id)
print(status)
# {
#   "tracking_id": "...",
#   "proposed_primitive": "retry_with_backoff",
#   "confidence": 0.72,
#   "action_taken": "auto_execute",
#   "execution_status": "success"
# }

# Register a custom repair primitive
from core.repair_primitives import RepairPrimitive, ValidationResult, ExecutionResult

class MyCustomRepair(RepairPrimitive):
    def validate(self, context):
        return ValidationResult(passed=True, reason="OK")
    def execute(self, context):
        # ... your repair logic ...
        return ExecutionResult(status="success", detail="Fixed it")

hc.register_primitive("my_custom_repair", MyCustomRepair())
```

## Diagnosis Pipeline

The seven-step pipeline processes every failure:

1. **Observe** — Embed failure event, store in DVS
2. **Recognize** — Activate NG-Lite substrate, detect novelty
3. **Recall** — Query DVS through substrate's learned topology
4. **Propose** — Select repair primitive based on recalled knowledge
5. **Validate** — Call primitive's `validate()` (MUST pass before execute)
6. **Execute** — If confidence >= threshold and validate passed
7. **Learn** — Store diagnostic chain, share via peer bridge

## Built-in Repair Primitives

| Primitive | Scope | Description |
|---|---|---|
| `process_restart` | host | SIGTERM → wait → verify |
| `cache_clear` | host | Remove cached data |
| `log_and_recommend` | any | Safe fallback — log and recommend |
| `config_adjust` | host | Modify config within safe bounds |
| `retry_with_backoff` | host | Retry with exponential backoff |
| `ng_lite_rebalance` | peer | Homeostatic weight scaling |
| `checkpoint_restore` | peer | Restore from known-good checkpoint |
| `connection_pool_reset` | host | Drain and reinitialize pool |

## Health Monitor

Background substrate health monitoring (runs as a daemon thread):

- **Weight divergence** — detects when synapses have learned extreme associations
- **Firing rate** — identifies dead nodes wasting substrate capacity
- **Novelty saturation** — flags when the substrate is losing discriminative power

Detected issues trigger proactive repairs through the diagnosis engine.

## Congregation

Peer deliberation for uncertain repairs. When confidence falls in the recommendation zone (0.40–0.70), the Congregation polls peer modules' shared learning events:

- Reads peer event files for repair records matching the failure embedding
- Similarity-weighted vote aggregation adjusts confidence up or down
- Consensus among 2+ peers can elevate a recommendation to auto-execute

## Pattern Compression

Periodic DVS compaction that clusters similar entries by embedding similarity:

- Triggered at 80% DVS fullness or every 7 days
- Clusters of 3+ similar entries are replaced with synthetic COMPRESSED_PATTERN entries
- Preserves aggregate knowledge (counts, outcomes, dominant primitive) while reducing storage

## Tier 3 Cluster Learning

Extends learning from dyadic peer pairs to full cluster knowledge sharing:

- Broadcasts successful repair outcomes to `shared_learning/` directory
- Syncs peer repair records into local DVS at startup and periodically
- Cluster-aggregated confidence blends local + all-peer success rates

## Configuration

Override defaults in `~/.et_modules/healing_collective/config.yaml`:

```yaml
confidence_auto_execute: 0.70
confidence_recommend: 0.40
confidence_host_premium: 0.15
repair_cooldown_seconds: 300
dvs_max_entries: 10000

health_monitor:
  enabled: true
  interval_seconds: 120
  weight_divergence_threshold: 2.0
  min_firing_rate: 0.001
  novelty_saturation_threshold: 0.95

congregation:
  confidence_spread: 0.15
  max_candidates: 4
  timeout_seconds: 10.0
  require_for_host_repairs: true

compression:
  trigger_pct: 0.80
  similarity_threshold: 0.85
  min_cluster: 3
  cycle_days: 7
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

AGPL-3.0 (see [NeuroGraph LICENSE](https://github.com/greatnorthernfishguy-hub/NeuroGraph))

## E-T Systems Ecosystem

Part of the E-T Systems module ecosystem:
- **NeuroGraph** — Dynamic Spiking Neuro-Hypergraph foundation
- **TrollGuard** — AI agent security pipeline
- **The Inference Difference** — Transparent inference routing proxy
- **The Healing Collective** — Self-healing intelligence (this module)
