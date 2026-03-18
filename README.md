
# XuanHu-v0.01

悬壶是一款聚焦C端就诊场景的Agents诊疗系统，旨在复刻临床医师的核心执业逻辑，打通动态沟通与精准决策双链路，区别于仅擅长静态医学问答的传统模型，实现全流程、策略化的辅助诊断服务。设计并落地了覆盖首程主动问诊、风险分诊、医学报告解读、个性化用药建议与健康科普的多Agent多模态智能体系。融合多模态感知、多智能体协作、双端检索采集与长效记忆完成系统编排，实现本地单卡高效推理部署与多病种场景验证。


2026/3/18: v0.01版本基于HuaTuo-VL与prompt agent开发，效果实在太差，尝试更换基模为doctor_r1。v0.1版本开发中ing

当前重点能力：

- 多 Agent 编排
- 首程问诊与分诊
- 医疗 RAG
- 长期记忆
- 检验报告与多模态扩展接口
- QLoRA / SFT / Agentic-RL 训练路线

## 快速开始

```bash
python -m medagent.main --question "我最近发烧咳嗽三天，需要怎么处理？"
python -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

## 目录结构

- `medagent/agents/`：各子 Agent
- `medagent/services/`：RAG、记忆、路由、安全与工具层
- `trainer/`：训练相关脚本，按数据、文本、视觉、RL 分层
- `scripts/`：部署、RAG 构建、评测和环境辅助脚本
- `data/`：样例数据和 benchmark 场景
- `docs/`：中文主文档索引

## 文档入口

- [文档总览](./docs/README.md)
- [系统架构](./docs/system/architecture.md)
- [训练总方案](./docs/training/training_playbook.md)
- [SFT / Agent / Adapter 定位](./docs/training/sft_adapter_strategy_zh.md)
- [BianQue-Intake SFT 说明](./docs/training/bianque_intake_sft_zh.md)
- [BianQue 首程问诊 Prompt](./docs/system/clinical_first_visit_prompt_zh.md)
- [RAG 设计](./docs/rag/rag_design.md)
- [HF 中文医疗语料审核报告](./docs/rag/hf_rag_audit_report_zh.md)
- [面试问答](./docs/interview/interview_qa.md)

## 当前模型路线

- 文本主干：`Qwen2.5-7B-Instruct`
- 视觉主干：`HuatuoGPT-Vision-7B-Qwen2.5VL`

当前判断是：

- 文本能力先围绕首程问诊、多 Agent、工具调用和长期记忆补强
- 视觉能力优先聚焦检验单 / 报告单 / 体检结果解释
- Agentic-RL 放在最后做

## 已完成的关键资产

- 中文 HF 医疗 RAG 底库与 chunk 流程
- HF 抽样审核报告（120 条）
- BianQue / LiShiZhen 双文本分工思路
- 视觉基模与训练路线决策
- AutoDL 同步与远端运行链路

## 后续重点

1. 完成 `BianQue-Intake` 专用 SFT 数据与训练
2. 清理剩余 placeholder 模块
3. 提升在线 RAG 检索效率
4. 补强 benchmark 和 patient simulator
5. 最后接 Agentic-RL
   =================
