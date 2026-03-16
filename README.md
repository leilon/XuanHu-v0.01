# MedAgent-7B 项目说明

这是一个面向互联网医疗场景的多 Agent 医疗助手项目骨架，当前重点对齐以下能力：

- 多 Agent 编排
- RAG 检索增强
- 长期记忆与持续学习
- 医疗安全兜底
- benchmark 横向评测
- 基于 7B 的 QLoRA / Agentic-RL 训练路线

## 快速开始

1. 准备 Python 3.10 及以上环境。
2. 运行示例：

```bash
python -m medagent.main --question "我最近发烧咳嗽三天，需要吃什么药？"
python -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

## 项目结构

- `medagent/config.py`：基础配置
- `medagent/orchestrator.py`：任务编排与多 Agent 工作流
- `medagent/agents/`：病情采集、分诊、问药、报告解读等子 Agent
- `medagent/services/`：RAG、长期记忆、安全守卫、工具层
- `medagent/benchmark/run.py`：benchmark 运行入口
- `data/benchmark_cases.json`：示例 benchmark 数据
- `docs/architecture.md`：系统架构设计
- `docs/training_playbook.md`：完整训练方案
- `docs/rag_design.md`：RAG 数据库设计
- `docs/interview_qa.md`：面试问答准备
- `docs/benchmark_plan.md`：benchmark 方案
- `docs/clinical_first_visit_prompt_zh.md`：临床首程问诊 base prompt

## 当前定位

这份代码不是最终线上系统，而是面向实习项目展示和后续训练扩展的开发骨架。后续需要逐步替换为：

- 真实 7B 推理后端
- 真实向量数据库与医学知识库
- 真实报告 OCR / 多模态解析链路
- 完整的 SFT / QLoRA / Agentic-RL 训练流程

## QLoRA 持续学习

训练一个任务 LoRA adapter，并自动注册到 adapter bank：

```bash
python scripts/train_qlora.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-file data/train_medical_sft.jsonl \
  --task general_intake \
  --dataset-name med_sft_v1 \
  --output-dir adapters/general_intake_v1
```

训练完成后，adapter 信息会写入 `adapters/index.json`。运行时会按任务选择 adapter，并融合：

- 结构化长期记忆
- 短期对话上下文
- 历史 episodic memory

## AutoDL 部署

本地 Linux/macOS：

```bash
bash scripts/deploy_to_autodl.sh <user> <host> <port> <remote_dir> [identity_file]
```

本地 Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_to_autodl.ps1 `
  -User <user> -Host <host> -Port <port> -RemoteDir <remote_dir> [-IdentityFile <key_path>]
```

## AutoDL 训练辅助命令

```bash
bash scripts/autodl_prepare_workspace.sh /root/autodl-tmp/medagent
bash scripts/autodl_download_assets.sh /root/autodl-tmp/medagent
export WANDB_API_KEY=xxx
bash scripts/run_sft_with_wandb.sh /root/autodl-tmp/medagent
python scripts/run_patient_sim_benchmark.py --scenarios data/sim_patient_cases.json
```

## 文档说明

主目录下的说明文档统一使用中文。英文版暂时归档在 `docs/en/`，仅作备份参考。
