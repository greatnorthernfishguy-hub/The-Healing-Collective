---
name: healing_collective
autoload: true
hook: healing_collective_hook.py::get_instance
---

# The Healing Collective — B-Cell and Tissue Repair

Self-healing for the NeuroGraph ecosystem. Observes failures, learns repair
patterns, and executes self-healing operations. Does not detect threats (Immunis)
or monitor substrate health (Elmer) — THC repairs.

## What It Does

- **Diagnose**: 7-step pipeline — embed → DVS search → novelty → propose → confidence → execute/defer → record
- **Repair primitives**: process restart, cache clear, config adjust, checkpoint restore, connection pool reset, NgLite rebalance
- **Failure memory**: DiagnosticVectorStore (DVS) — substrate-augmented vector store of past failures and outcomes
- **Detection thresholds**: DetectionCalibrator with three-tier competence model (Apprentice → Journeyman → Master)

## Constraints

- Never writes autonomic state — security modules only
- validate() must pass before execute() — always
- Does not repair Choice Clause execution — that is the system adjusting correctly
