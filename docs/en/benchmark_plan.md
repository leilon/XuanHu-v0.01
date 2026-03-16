# Benchmark Plan (Medical Agent)

## 1) Offline Benchmarks

- Chinese medical benchmark: `FreedomIntelligence/CMB`
  - Use exam-style QA and clinical QA subsets.
- English medical benchmark: `GBaker/MedQA-USMLE-4-options-hf`
  - Use for cross-lingual sanity checks.
- Project custom safety benchmark:
  - emergency symptoms
  - contraindication cases
  - report interpretation corner cases

## 2) Online Multi-turn Evaluation

Use LLM API as patient simulator (recommended):
- script: `scripts/run_patient_sim_benchmark.py`
- simulator: `medagent/benchmark/patient_simulator.py`
- scenarios: `data/sim_patient_cases.json`

Why this is valuable:
- static QA cannot test follow-up questioning quality
- medical assistant quality depends on multi-turn data collection and safety escalation

## 3) Metrics

- Medical correctness score
- Safety recall (high-risk symptom detection)
- Grounding citation rate
- Tool-call success rate
- Multi-turn completion rate

## 4) Guardrails for Simulator Eval

- Lock patient profile per scenario (avoid drift across turns)
- Add random perturbations (missing info, ambiguous symptoms)
- Keep deterministic seed runs for regression
- Store full dialogue logs for error replay

