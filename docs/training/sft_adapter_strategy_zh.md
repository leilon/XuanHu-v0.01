# SFT / Agent / Adapter 定位

这份文档专门用来把当前项目里最容易混淆的三个概念拆开：

- `Agent`
- `SFT 数据`
- `Adapter`

## 1. 三者分别是什么

### 1.1 Agent

`Agent` 是运行时角色，是 `QiBo-Orchestrator` 编排出来的一段职责。

例如：
- `BianQue-Intake`：首程问诊、补病史、识别红旗
- `LiShiZhen-Education`：科普解释、患者友好说明
- `CangGong-Report`：报告理解、异常项解释
- `ShenNong-Medication`：药物问答、禁忌检查

Agent 是“系统分工”，不等于“必须各训一个模型”。

### 1.2 SFT 数据

`SFT 数据` 是监督信号，是为了教模型学会某一类行为而清洗出来的任务数据。

例如：
- 首程短轮次追问数据
- 长解释型科普数据
- 报告图像解释数据

同一个模型可以吃多份不同风格的 SFT 数据，但如果任务风格差异太大，就应该拆开。

### 1.3 Adapter

`Adapter` 是挂在基座模型上的参数增量，是训练产物。

它解决的是：
- 不同任务要不要分开专门适配
- 能不能低成本持续迭代

所以：
- `Agent` 是运行时角色
- `SFT 数据` 是训练输入
- `Adapter` 是训练结果

三者不是一回事。

## 2. 当前项目推荐映射

### 2.1 文本基座

- 基座模型：`Qwen2.5-7B-Instruct`

### 2.2 视觉基座

- 基座模型：`HuatuoGPT-Vision-7B-Qwen2.5VL`

### 2.3 第一阶段只训 3 条主线

1. `BianQue-Intake` adapter
- 面向首程问诊
- 只吃短轮次、一问一答、补病史风格数据

2. `LiShiZhen-Education` adapter
- 面向科普与解释
- 吃长解释、指标释义、安抚说明

3. `CangGong-Report` adapter
- 面向视觉报告解释
- 吃图像 + 提问 + 异常解释 / 下一步建议

### 2.4 暂时不单独训 adapter 的角色

第一版 demo 里，这几个角色不急着单独训：

- `QiBo-Orchestrator`
- `HuaTuo-Triage`
- `ShenNong-Medication`
- `SiMiao-Memory`

原因不是它们不重要，而是它们更依赖：
- 工作流编排
- RAG
- 规则
- 工具调用
- 长期记忆

现在先把这些角色做成“基座模型 + prompt + 工具 + RAG”的系统能力，更符合两天内做出可讲 demo 的目标。

## 3. 数据路径怎么约定

原始下载数据：
- `/root/autodl-tmp/medagent/datasets/sft`
- `/root/autodl-tmp/medagent/datasets/rl`

清洗后的任务数据：
- `/root/autodl-tmp/medagent/datasets/curated/bianque_intake`
- `/root/autodl-tmp/medagent/datasets/curated/lishizhen_education`
- `/root/autodl-tmp/medagent/datasets/curated/canggong_vision`

训练产物：
- `/root/autodl-tmp/medagent/outputs/adapters`

这条边界很重要：
- `datasets/sft` 是原始下载区
- `datasets/curated/...` 才是可以直接训练的任务数据

## 4. 推荐训练顺序

1. 先清洗 `BianQue-Intake` 数据
2. 训 `BianQue-Intake` adapter
3. 再清洗 `LiShiZhen-Education` 数据
4. 训 `LiShiZhen-Education` adapter
5. 再处理视觉报告数据，训 `CangGong-Report`
6. 最后再考虑 `Agentic-RL`

## 5. 为什么不建议“一 agent 对应一个 adapter”

因为很多 agent 的差异不在语言风格，而在系统能力。

例如：
- `ShenNong-Medication` 的上限主要来自药物知识库和工具
- `QiBo-Orchestrator` 的上限主要来自路由和编排
- `SiMiao-Memory` 的上限主要来自记忆写入、检索和冲突处理

如果一开始把它们全拆成独立 adapter：
- 训练目标会发散
- 数据会不够纯
- 维护成本会明显上升

对现在的 demo，更稳的做法是：
- 只把“语言行为差异特别大”的部分拆成 adapter
- 其余先依赖系统设计解决

## 6. 当前一句话方案

当前最清晰的方案是：

- 一个文本基座
- 一个视觉基座
- 三个核心 adapter
- 其余 agent 先靠编排、RAG、记忆和工具支撑

这样定位清楚，也最适合后面面试时讲清楚“为什么这样设计”。
