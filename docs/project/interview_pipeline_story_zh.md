# 面试讲述主线

## 1. 我们现在的路线

当前项目的核心路线是：

1. 选择一个**公开可复现路线更完整**的医学多模态基模  
2. 在它上面搭建 `LangGraph + Prompt-based Multi-Agent + Dual RAG + Memory`
3. 先把 demo 主流程跑通
4. 再用多轮 benchmark 去观察系统行为
5. 最后再做 synthetic agentic-RL，比较前后性能

这条路线更适合面试，因为它强调：
- 你知道如何做系统设计
- 你知道如何分阶段推进
- 你不会为了追求“最强模型名头”而牺牲可解释性和可复现性

## 2. 为什么当前不建议切到 HealthGPT

### 2.1 当前推荐保留 HuatuoGPT-Vision-7B-Qwen2.5VL

推荐保留：
- `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

原因：
- 它和 `Qwen2.5-VL` 路线兼容
- 官方明确给出了从 `Qwen2.5-VL` 出发的训练流程
- 官方给了训练阶段划分：初始化、视觉对齐、视觉指令微调
- 官方给了评测集和命令
- 我们已经在 AutoDL 上把它跑起来了

### 2.2 HealthGPT 很强，但不适合当前面试项目主线

`HealthGPT` 的优势：
- 学术上很强
- 医学多模态能力看起来很亮眼
- 结构设计也很有意思，例如 H-LoRA、硬路由等

但对我们当前项目有两个明显问题：

1. 它的公开训练链路还不完整  
- 官方仓库里写了 `Release training scripts` 仍在 TODO
- 当前公开内容更偏：
  - inference
  - 部分权重
  - 部分 adapter / H-LoRA 权重

2. 它的底座路线和我们当前系统衔接不如 Huatuo/Qwen 自然  
- `HealthGPT-M3` 基于 `Phi-3-mini-4k-instruct`
- `HealthGPT-L14` 基于 `Phi-4`
- 还有单独视觉编码器、H-LoRA、fusion layers、VQGAN 依赖
- 这会让你面试里很难讲成一个清晰、可控、工程上自然的故事

### 2.3 结论

对于当前这个面试项目：

- **不要切到 HealthGPT 作为主基模**
- **保留 HuatuoGPT-Vision-7B-Qwen2.5VL**
- 如果要提 HealthGPT，把它当成“调研过的更强候选路线”即可

## 3. 面试里怎么讲基模来源

建议**主动说**，不要故意略去。

推荐讲法：

> 当前项目的视觉基模采用 Hugging Face 上 `FreedomIntelligence` 发布的 `HuatuoGPT-Vision-7B-Qwen2.5VL`。  
> 它本质上是一个基于 `Qwen2.5-VL` 的医学多模态模型。  
> 我的工作重点不在于重新从零预训练一个基模，而在于把基模组织成一个面向 C 端就诊流程的多 agent 系统，并补齐 RAG、长期记忆、benchmark 和后续 agentic-RL 的闭环。

这比“微妙略去”更稳，也更专业。

## 4. 当前最推荐的完整流程

### 阶段 A：基模选择与说明

叙事上这样讲：

1. 原始通用架构路线来自 `Qwen2.5-VL`
2. 当前直接采用其医学适配版本 `HuatuoGPT-Vision-7B-Qwen2.5VL`
3. 这样做是为了把时间和算力集中到：
- agent 编排
- RAG
- memory
- benchmark
- synthetic agentic-RL

### 阶段 B：系统搭建

当前系统结构：

- `LangGraph` 负责多 agent 编排
- `BianQue-Intake` 负责首程问诊
- `HuaTuo-Triage` 负责分诊
- `ShenNong-Medication` 负责用药建议
- `CangGong-Report` 负责报告解读
- `LiShiZhen-Education` 负责医学解释
- `SiMiao-Memory` 负责长期记忆提取和读取

### 阶段 C：Dual RAG

RAG 分两侧：

1. 用户端
- 最近对话
- 年龄、性别
- 既往史
- 过敏史
- 长期用药
- 历史 episode

2. 资料端
- 中文医疗问答语料
- 中文医疗知识语料
- 后续会继续向正式知识库升级

### 阶段 D：多轮 benchmark

这里是下一步最重要的工作。

建议做两类 benchmark：

#### 1. 现成 benchmark

用于单轮和基础能力对比：
- `CMB`
- `MedQA`
- `HuatuoGPT-Vision` 官方多模态评测集

价值：
- 证明底座和基础回答能力没有太差

#### 2. 自建多轮 benchmark

这部分最重要。

建议用强 LLM API 构造多轮 benchmark，重点不是让 API 帮我们回答，而是让它：

- 扮演病人
- 扮演 judge
- 清洗 benchmark case

推荐场景：
- 呼吸道发热
- 胸痛胸闷
- 腹痛腹泻
- 妇产/妊娠相关
- 儿科发热/惊厥
- 神经系统急症
- 报告解读与复诊

每条 case 应包括：
- hidden case summary
- 必问槽位
- red flags
- 正确分诊标签
- 标准下一步检查建议
- 用户的口语化表达

### 阶段 E：synthetic agentic-RL

这是最后一步，不是第一步。

建议顺序：

1. 先测没有 agentic-RL 时的多轮 benchmark 表现
2. 再合成 agentic-RL 数据
3. 再比较前后变化

agentic-RL 数据建议：
- `chosen`: 正确追问、识别红旗、分诊合理、用药保守
- `rejected`: 漏关键病史、漏高危信号、分诊过松、给出不安全建议

## 5. 多轮 benchmark 怎么构造

建议用强 API 做三件事：

1. **病人模拟器**
- 根据 hidden case 扮演病人
- 只在被问到时逐步透露信息
- 不主动一次性把病史说完

2. **judge**
- 判断系统有没有问到关键槽位
- 有没有识别 red flag
- 分诊是不是对
- 有没有给出危险建议

3. **case cleaner**
- 把 benchmark case 改写成更像真实中文患者表达

这样做的好处：
- 你不用一开始就自己合成完整 RL 训练集
- 先把 benchmark 搭起来，评估闭环先成立
- 后续 benchmark case 还可以回流成 agentic-RL 数据

## 6. synthetic agentic-RL 前后应该比什么

最值得比较的是：

1. 必问槽位命中率
2. 红旗症状召回率
3. 分诊准确率
4. 平均追问轮数
5. 不安全建议率
6. RAG 引用命中率
7. 长期记忆命中率

推荐对比版本：

1. `Base Model Only`
- 只给基模和基础 prompt

2. `Prompt Agents + RAG + Memory`
- 当前系统版本

3. `Prompt Agents + RAG + Memory + Agentic-RL`
- 后续强化版本

## 7. 面试里最好的说法

推荐这样讲：

> 这个项目我没有追求“从零训练一个医学基模”，而是先选择了一个公开训练路线更清晰、已经过医学适配的多模态基模，基于它搭建面向 C 端就诊流程的 agent 系统。  
> 我把重点放在多 agent 编排、长期记忆、双通道 RAG、多轮 benchmark 和后续 agentic-RL 闭环上。  
> 这样做的好处是，系统价值更清晰，也更适合快速验证真实问诊场景下的行为改进。

## 8. 当前文档支持情况

当前已经有的文档：

- 架构说明
- lightweight demo 路线
- Huatuo 系列调研
- RAG 设计
- agentic-RL 合成思路
- benchmark 计划

但还缺一类最关键的文档：

- **把“基模选择 -> agent 构建 -> benchmark -> agentic-RL 对比”串成一条面试叙事主线的文档**

本文档就是为这个目的补上的。
