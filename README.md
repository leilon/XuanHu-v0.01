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
python -m medagent.main --question "我最近发烧咳嗽三天，需要吃什么药？"
python -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

## Project Layout

- `medagent/config.py`: app and routing config
- `medagent/orchestrator.py`: planner + agent workflow
- `medagent/agents/`: specialized agents
- `medagent/services/`: RAG, memory, safety, and tool abstraction
- `medagent/benchmark/run.py`: benchmark and horizontal comparison
- `data/benchmark_cases.json`: toy benchmark set
- `docs/architecture.md`: system architecture design
- `docs/training_playbook.md`: end-to-end training plan (AutoDL + W&B)
- `docs/rag_design.md`: RAG database design
- `docs/interview_qa.md`: interview Q&A prep
- `docs/benchmark_plan.md`: benchmark strategy + patient simulator evaluation

## Notes

This is a development scaffold for internship project demonstration. Replace stubs with:

- A real 7B inference backend (vLLM/Transformers)
- Real vector DB + medical knowledge base
- Real report OCR/multimodal pipeline
- SFT/RL/QLoRA training pipelines

## QLoRA Continual Learning

Train a task adapter (and auto-register into adapter bank):

```bash
python scripts/train_qlora.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-file data/train_medical_sft.jsonl \
  --task general_intake \
  --dataset-name med_sft_v1 \
  --output-dir adapters/general_intake_v1
```

After training, adapter metadata is recorded in `adapters/index.json`. Runtime picks an adapter by task
and fuses:

- profile memory (`long_term`)
- short context (`short_term`)
- episodic memory (`episodic`)

## Deploy To AutoDL

Linux/macOS local machine:

```bash
bash scripts/deploy_to_autodl.sh <user> <host> <port> <remote_dir> [identity_file]
```

Windows PowerShell local machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_autodl.ps1 `
  -User <user> -Host <host> -Port <port> -RemoteDir <remote_dir> [-IdentityFile <key_path>]
```

## AutoDL Training Helpers

```bash
bash scripts/autodl_prepare_workspace.sh /root/autodl-tmp/medagent
bash scripts/autodl_download_assets.sh /root/autodl-tmp/medagent
export WANDB_API_KEY=xxx
bash scripts/run_sft_with_wandb.sh /root/autodl-tmp/medagent
python scripts/run_patient_sim_benchmark.py --scenarios data/sim_patient_cases.json
```
