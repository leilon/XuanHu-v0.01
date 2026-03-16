# MedAgent-7B Scaffold

This repository contains a runnable scaffold for a multi-agent medical assistant aligned with:

- Agentic orchestration
- RAG integration
- Memory (short-term + long-term interface)
- Safety guardrails
- Benchmark-based horizontal comparison

## Quickstart

1. Create a Python 3.10+ environment.
2. Run:

```bash
python -m medagent.main --question "I have had fever and cough for three days, what should I do?"
python -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

## Project Layout

- `medagent/config.py`: app and routing config
- `medagent/orchestrator.py`: planner and agent workflow
- `medagent/agents/`: specialized agents
- `medagent/services/`: RAG, memory, safety, and tool abstraction
- `medagent/benchmark/run.py`: benchmark and horizontal comparison
- `data/benchmark_cases.json`: toy benchmark set

## Notes

This is a development scaffold for internship project demonstration. Replace stubs with:

- A real 7B inference backend
- A real vector DB and medical knowledge base
- A real report OCR / multimodal pipeline
- SFT / RL / QLoRA training pipelines
