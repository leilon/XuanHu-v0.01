# XuanHu-v0.01

`悬壶` 是一个面向 C 端就诊场景的医疗 Agent 系统实验项目，目标是把首程问诊、风险分诊、资料检索、长期记忆和自动首程病历串成一条可运行的工程链路。

当前仓库保留的是 `v0.01` 技术路线归档版本：
- 多 Agent 编排
- 首程问诊与分诊
- 医疗 RAG
- 长期记忆
- 报告解读与多模态接口
- benchmark / patient simulator / 后续 Agentic-RL 预留

> 说明：`v0.01` 曾基于 `HuatuoGPT-Vision-7B-Qwen2.5VL + prompt agents` 做快速验证。我们已确认这条路线链路可跑通，但模型能力不足，后续版本将转向以 `Doctor-R1` 为核心的重构路线。

## 当前状态

- 本仓库当前更适合当作：
  - 系统链路归档
  - RAG / benchmark / 多轮对话设计参考
  - 后续 `Doctor-R1` 路线的工程底座
- 不建议把 `v0.01` 视为最终模型效果版本

## 核心文档

- [文档总览](./docs/README.md)
- [多轮 10 Case 摘要](./docs/reports/multiturn_casebook_summary_zh.md)
- [RAG 向量化效率报告](./docs/reports/rag_latency_report_zh.md)
- [简历指标快照](./docs/project/resume_metrics_snapshot_zh.md)
- [HealthBench 接入说明](./docs/evaluation/healthbench_integration_zh.md)

## 使用过的基模

- [FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL)
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## 使用过的主要数据集

### 文本 / 医疗问答
- [FreedomIntelligence/HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1)
- [FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)
- [FreedomIntelligence/huatuo_encyclopedia_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa)
- [FreedomIntelligence/huatuo_knowledge_graph_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_knowledge_graph_qa)
- [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- [wangrongsheng/cMedQA-V2.0](https://huggingface.co/datasets/wangrongsheng/cMedQA-V2.0)
- [BillGPT/Chinese-medical-dialogue-data](https://huggingface.co/datasets/BillGPT/Chinese-medical-dialogue-data)

### 多模态 / 医学视觉参考
- [FreedomIntelligence/PubMedVision](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision)

## 当前仓库结构

- `medagent/agents/`：运行时 agent
- `medagent/services/`：RAG、记忆、路由、安全、向量检索、多轮状态
- `trainer/`：训练与数据准备脚本
- `scripts/`：部署、评测、索引构建、环境辅助脚本
- `data/`：casebook、benchmark 场景、示例数据
- `docs/`：中文主文档与报告

## 当前可复用资产

- 中文医疗 RAG 语料整理与向量化流程
- 多轮患者模拟器与 10 case 测试集
- RAG 向量化前后效率对比
- LangGraph 多轮问诊状态机雏形
- 自动首程摘要与结构化病历产出逻辑

## 下一步方向

- 以 [Doctor-R1](https://github.com/thu-unicorn/Doctor-R1) 为核心重构门诊专家 Agent
- 保留现有 RAG 语料、benchmark 和病历存储设计
- 将高风险分诊、初步诊断和首程病历生成做成更稳定的混合系统
