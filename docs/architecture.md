# 多Agent医疗助手主体架构（7B微调版）

## 1. 目标

- 面向C端用户的诊前-诊中-诊后一体化智能问诊体验。
- 支持多轮对话、工具调用、可追溯证据与安全兜底。
- 为后续 7B 模型 SFT + Agentic RL + QLoRA 持续学习预留训练接口。

## 2. 核心组件

### 2.1 Orchestrator Agent

- 输入：用户问题、历史会话、长期记忆画像。
- 输出：任务计划（Task Graph）与最终回复。
- 关键能力：任务拆解、动态路由、失败重试与兜底策略。

### 2.2 Specialized Agents

- Intake Agent：病情采集与缺失信息追问。
- Triage Agent：风险分层与就医时效建议。
- Medication Agent：对症建议与相互作用检查。
- Report Agent：报告文本/影像解析后的异常摘要与解释。

### 2.3 Service Layer

- RAG：检索指南/药典/科普文档，提供可引用证据。
- Memory：短期记忆（会话窗口）+长期记忆（用户病史、过敏、偏好）。
- Safety Guard：高风险症状拦截与强制安全提示。
- Tools：药物相互作用、报告解析、院内流程查询等能力。

## 3. 多Agent执行流

1. 用户输入进入 Orchestrator。
2. Safety Guard 先做风险判别。
3. Orchestrator 生成任务序列：`intake -> triage -> report(optional) -> medication -> rag_summary`。
4. 各子Agent独立执行并回传结构化结果。
5. RAG附上证据来源，Safety Guard做最终拦截。
6. 生成带引用的最终答复，并写入记忆。

## 4. 与训练框架衔接

- SFT数据：对话样本、任务分解轨迹、工具调用标签。
- Agentic RL：
  - 状态：历史对话、任务图、风险等级、检索证据质量。
  - 动作：追问、调用工具、给出建议、转人工。
  - 奖励：医学准确性、任务完成率、安全性、用户满意度。
- QLoRA持续学习：
  - 每轮数据增量训练 LoRA Adapter。
  - 通过经验回放降低遗忘。
  - 支持按任务加载 Adapter Bank。

## 5. Benchmark横向对比

- 对比对象：
  - Baseline（单Agent直接回复）
  - Multi-Agent（编排器+工具+RAG）
- 评测维度：
  - 医学任务完成度
  - 安全合规性
  - 证据可追溯性
- 输出：
  - 单Case得分
  - 策略级平均分（用于模型版本回归）

## 6. 已落地的QLoRA长期记忆能力

- Adapter Bank: `medagent/services/adapter_bank.py`
  - 管理不同任务LoRA适配器（问诊、问药、报告解读）。
  - 将训练产物登记到 `adapters/index.json`，支持运行时路由。
- 外部长期记忆: `medagent/services/memory.py`
  - profile memory（结构化事实）
  - episodic memory（历史对话片段）
- 记忆融合推理: `medagent/services/memory_fusion.py`
  - 运行时融合 profile + recent turns + episodic memory
  - 自动按任务选择adapter（若存在）
- QLoRA训练入口: `scripts/train_qlora.py`
  - 支持增量训练与回放文件（replay）实现持续学习
