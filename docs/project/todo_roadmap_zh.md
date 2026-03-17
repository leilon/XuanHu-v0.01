# 当前待办清单

## 1. Git 与同步

- 保持本地仓库和 AutoDL 工作目录一致
- 后续需要时再接入 GitHub 正式仓库

## 2. 模型与环境

- 当前文本主干：`Qwen2.5-7B-Instruct`
- 当前视觉主干：`HuatuoGPT-Vision-7B-Qwen2.5VL`
- 需要确认哪些基模还没下完，哪些可以删掉
- 等脚本稳定后再切换到 4 卡 `5090` 的 AutoDL 容器

## 3. 首程问诊主线

- 重构 `clinical_pathway.py`
- 从规则模板升级为：
  - 症状分类
  - 问诊树
  - 红旗规则
  - 分诊规则
- 当前 `IntentRouter` 仍然是规则型占位实现，后续要升级
- 所有首程病例都要从“信息不完整的真实主诉”开始
- 不能默认用户会主动说出体温、持续时间、血压等关键数值

## 4. 长期记忆与遗忘

- 拆分两层：
  - 医学知识 RAG
  - 患者长期记忆
- 明确长期存什么、短期存什么、什么需要遗忘
- 把结构化首程摘要纳入长期记忆设计
- 首程摘要优先由强 API 生成，而不是只靠本地 7B

## 5. API 路由

- 设计统一的 `LLMRouter`
- 高风险、低置信、多工具链、复杂视觉病例升级到强 API
- 强 API 同时承担：
  - 首程摘要生成
  - synthetic judge
  - patient simulator

## 6. RAG 主线

- 保留中文 HF 医疗语料主线
- 移除 MSD 中文专业版
- 完成纯 HF 版本的 chunk 重建
- 补 embedding / reranker / 向量索引
- 输出一份 HF 抓取样本审核报告，至少 100 条

## 7. QLoRA 主线

- 重新设计文本 SFT 数据桶：
  - `BianQue-Intake`：短轮次、一问一答、首轮追问风格
  - `LiShiZhen-Education`：长解释、科普、指标释义、安抚
- 不把 QLoRA 当 test-time training 主工具
- 更适合做：
  - task adapter
  - 周期性持续学习
  - teacher 蒸馏

## 8. Agentic-RL 主线

- 暂时放在最后
- 先把首程问诊、RAG、长期记忆、多模态打稳
- 后面再做 trajectory preference 和模拟器评测

## 9. 仍然偏占位符的高优先级模块

- `medagent/services/rag.py`
- `medagent/services/tools.py`
- `medagent/agents/report.py`
- `medagent/agents/medication.py`
- `medagent/services/intent_router.py`
- `medagent/benchmark/evaluator.py`
- `medagent/benchmark/run.py`

## 10. 面试展示物

- 架构图
- 首程问诊流程图
- RAG 语料说明
- 训练路线图
- 面试问答稿
