# 医疗 Agentic-RL 数据调研

## 结论先说

现阶段几乎没有一份“大规模、干净、直接可用”的公开医疗 agentic-RL 偏好数据集，可以像通用 `UltraFeedback` 那样直接拿来训练多轮医疗 agent。

更靠谱的路线是三段式：

1. 先用高质量医疗 SFT 数据把 `Qwen2.5-7B-Instruct` 训稳。
2. 再用少量真实医疗 preference / reward 数据做偏好热启动。
3. 最后用医疗 benchmark、病例脚本和多轮模拟环境合成 agentic preference pairs。

## 直接可用但规模有限的数据

### `WhiteF4lcon/medical-rlhf-pairs`

链接：[WhiteF4lcon/medical-rlhf-pairs](https://huggingface.co/datasets/WhiteF4lcon/medical-rlhf-pairs)

特点：
- 字段简单：`instruction / positive / negative`
- 约 `5.9k` 条
- 适合做第一版 DPO 热启动

我抽样看到的特点：
- 正样本通常是完整的医学解释
- 负样本很多是通过插入随机医学词制造的“坏回答”

优点：
- 格式标准，接入最方便
- 至少是医疗域，不是纯通用偏好数据

缺点：
- 不是真正的多轮 agent 轨迹
- 负样本偏“语言损坏”，不是“临床流程错误”

建议用途：
- 只做 RL 种子，不做主力 agentic-RL 数据

### `drewli20200316/medical-rlhf-data`

链接：[drewli20200316/medical-rlhf-data](https://huggingface.co/datasets/drewli20200316/medical-rlhf-data)

特点：
- 同时包含 `sft_*` 和 `rm_*` 文件
- 主要字段是 `prompt / chosen / rejected`
- 更像传统医学 QA 偏好数据

我抽样看到的特点：
- 大多是长医学问答
- `chosen` 和 `rejected` 往往都比较像正确答案，差异偏细

优点：
- 格式规范
- 可以直接转 DPO

缺点：
- 更偏静态医学问答
- 几乎没有工具调用、追问、分诊升级这类 agent 行为

建议用途：
- 做 reward / DPO 辅助集
- 不建议单独拿来代表 agentic-RL

### `LangAGI-Lab/Medical_reward_bench`

链接：[LangAGI-Lab/Medical_reward_bench](https://huggingface.co/datasets/LangAGI-Lab/Medical_reward_bench)

特点：
- 约 `2.3k` 条
- 含 `input / chosen / reject / criteria / chosen_model / reject_model`
- 更像“医学奖励模型评测集”

我抽样看到的特点：
- 数据多来自 `medqa`
- 更适合分析模型间偏好差异和打分维度

优点：
- 自带 `criteria`，方便做 reward 建模

缺点：
- 规模偏小
- 仍然不是多轮 agent 轨迹

建议用途：
- 适合做 `judge` 或 reward model 的验证集
- 不建议作为主要策略优化数据

## 最值得改造的数据

### `katielink/agentclinic_medqa`

链接：[katielink/agentclinic_medqa](https://huggingface.co/datasets/katielink/agentclinic_medqa)

它不是标准 preference dataset，但非常适合改造成 agentic-RL 数据。

原因：
- 每条样本本身就是一个完整 OSCE 病例包
- 包含：
  - `Objective_for_Doctor`
  - `Patient_Actor`
  - `Physical_Examination_Findings`
  - `Test_Results`
  - `Correct_Diagnosis`

这意味着我们可以从一条病例派生出：
- 病人首轮提问
- 必问病史清单
- 应该调用的工具
- 正确升级路径
- 正确最终建议
- 多种失败模式负样本

它特别适合改造成我们现在的四段式合成流程：
- `case packet`
- `expert trajectory`
- `failure trajectory`
- `judge filtering`

### `DoctorAgent-RL`

链接：[DoctorAgent-RL](https://github.com/AI-in-Health/DoctorAgent-RL)

价值：
- 方向直接对齐“医疗 RL / medical agent”
- 更接近我们目标任务，不是单纯问答

我当前判断：
- 即便不直接提供大规模 pairwise 数据，也很适合拿来参考：
  - 任务定义
  - 状态空间
  - 追问策略
  - 奖励设计
  - 医疗环境模拟

建议用途：
- 作为 agent 训练流程和奖励设计参考
- 如果仓库提供轨迹或 simulator，可以转成 preference pairs

### `MedAgentBench`

链接：[MedAgentBench](https://github.com/gersteinlab/MedAgentBench)

价值：
- 更像真实临床环境 benchmark
- 可以用于采集 agent 在复杂病例上的轨迹

建议用途：
- 主要拿来做评测和轨迹蒸馏
- 从 benchmark run 中截取 `good / bad trajectories` 做偏好对

### `AgentClinic`

链接：[AgentClinic](https://github.com/gersteinlab/AgentClinic)

价值：
- 很适合拿来做“LLM 扮演病人”的环境
- 可用于多轮对话 benchmark

建议用途：
- 评测优先
- 其次用于生成多轮病例轨迹

## 不建议直接当主力 RL 数据的数据

### `Julian2002/Medical-LM-32B-Preference`

链接：[Julian2002/Medical-LM-32B-Preference](https://huggingface.co/datasets/Julian2002/Medical-LM-32B-Preference)

虽然规模有 `46k+`，但我抽样发现里面混入了明显不安全或非医疗内容，比如：
- 询问如何篡改过期牛奶日期
- 打听某个具体住址搬来的人是谁

这说明：
- 名字里有 `Medical`，不代表数据真的纯医疗
- 不能直接拿来训练

结论：
- 直接排除

### `saepark/ultrafeedback-binarized-preferences-medical-cldfilter-*`

链接：[saepark medical-cldfilter](https://huggingface.co/datasets/saepark/ultrafeedback-binarized-preferences-medical-cldfilter-train)

之前我们已经验证过，这类数据的主要问题是：
- 过滤后仍有不少非医疗样本
- 很多样本不是 agent 行为差异，而是普通回答风格差异

结论：
- 可以做弱辅助
- 不要把它当核心 agentic-RL 数据

## 和 7B 最适配的数据方案

### SFT 主料

推荐主干：
- `FreedomIntelligence/HuatuoGPT-sft-data-v1`
- `shibing624/medical` 中过滤后的高质量中文子集
- `BillGPT/Chinese-medical-dialogue-data`
- `wangrongsheng/cMedQA-V2.0`
- `FreedomIntelligence/Medical-R1-Distill-Data-Chinese`

建议做法：
- 先过滤，再混合
- 不要盲目追求百万条
- 对 7B 来说，`20万 - 50万` 条高质量中文医疗样本通常比粗糙的百万样本更稳

### Agentic-RL 主料

推荐主干：
- `agentclinic_medqa` 这类病例包
- benchmark / simulator 中跑出来的多轮轨迹
- 你自己系统的多 agent 日志
- 少量公开 medical RLHF 数据作为热启动

推荐配比：
- `60%` 合成的 agentic preference pairs
- `20%` 真实多轮日志回流
- `10%` 小规模公开 medical preference
- `10%` 人工构造高风险负样本

## 我们项目里最靠谱的落地方案

### 第一阶段

用高质量 SFT 把文本 7B 训稳：
- 症状采集
- 分诊
- 问药
- 医疗科普
- 报告解释

### 第二阶段

用以下数据做 agentic preference 种子：
- `WhiteF4lcon/medical-rlhf-pairs`
- `drewli20200316/medical-rlhf-data`
- `LangAGI-Lab/Medical_reward_bench`

### 第三阶段

以 `agentclinic_medqa + benchmark case + 系统日志` 为种子，批量合成：
- `chosen`: 正确追问、合理调用工具、给出升级建议
- `rejected`: 提前下结论、漏问过敏史、漏掉危险信号、不给证据、乱用药

### 第四阶段

用强模型当：
- teacher
- judge
- patient simulator

本地 7B 当：
- 主力医疗 agent
- 低成本在线推理模型

## 为什么这比“直接找现成 RL 数据”更靠谱

因为医疗 agent 的核心不是“谁回答得更像教科书”，而是：
- 有没有追问关键病史
- 有没有识别红旗症状
- 有没有调用正确工具
- 有没有给出安全升级路径
- 有没有把结论和证据对上

公开偏好数据往往只能监督“回答文本”，很难监督“完整 agent 行为”。

所以我们真正该训练的是：
- `multi-turn behavior`
- `tool-use policy`
- `safety escalation policy`
- `evidence-grounded response policy`

这也是为什么我们当前的合成脚本要做成：
- 隐藏病例包
- 专家轨迹
- 失败模式负样本
- judge margin 过滤

而不是简单地“让大模型写两个答案然后二选一”。
