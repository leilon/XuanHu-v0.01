# HealthBench 接入说明

## 1. HealthBench 是什么

`HealthBench` 是 OpenAI 在 2025 年公开的医疗评测基准。

它的特点不是传统医学考试题，而是：

- `5000` 条真实感更强的医疗对话
- 多轮、多语言
- 每条样本都有医生编写的 rubric
- 最终分数不是简单字符串匹配，而是按 rubric 打分

它还有两个常见子集：

- `healthbench_consensus`
- `healthbench_hard`

## 2. 官方代码在哪里

官方参考实现放在：

- `openai/simple-evals`

官方说明里明确写了：

- `simple-evals` 会继续保留 `HealthBench`、`BrowseComp`、`SimpleQA` 的参考实现
- `HealthBench` 的官方调用入口在 `healthbench_eval.py` 和 `simple_evals.py`

## 3. 官方调用方式

官方最直接的运行命令是：

```bash
python -m simple-evals.simple_evals --eval=healthbench --model=gpt-4.1
python -m simple-evals.simple_evals --eval=healthbench_consensus --model=gpt-4.1
python -m simple-evals.simple_evals --eval=healthbench_hard --model=gpt-4.1
```

也可以加这些参数：

```bash
python -m simple-evals.simple_evals \
  --eval=healthbench \
  --model=gpt-4.1 \
  --examples 100 \
  --n-threads 32 \
  --n-repeats 1
```

其中：

- `--examples` 控制抽样条数
- `--n-threads` 控制 HealthBench 的并发评分线程数
- `--n-repeats` 控制重复采样次数

## 4. 官方数据从哪里来

`healthbench_eval.py` 里直接写了公开数据地址：

- 主集 JSONL
- `hard` 子集 JSONL
- `consensus` 子集 JSONL

也就是说，官方实现本身会从公开 blob 地址读取数据，不需要你自己先手工整理成别的格式。

## 5. 官方是怎么打分的

官方实现里，每条样本会包含：

- 一个多轮对话上下文
- 一组 rubric 条目
- 每个 rubric 条目有分值

然后用一个 `grader model` 去判断：

- 当前回答是否满足某条 rubric

最后按正负分条目聚合成总体分数。

当前官方参考实现里，默认 grader 是：

- `gpt-4.1`

所以 `HealthBench` 本质上是：

- `多轮医疗对话 + rubric-based grading + model-as-judge`

## 6. 它和我们项目怎么结合

### 6.1 能直接用的地方

最适合先用在：

- 评估我们系统的最终回答质量
- 看安全性、相关性、上下文补充是否合理

尤其适合拿来比较：

1. `基模裸答`
2. `青囊智诊 + LangGraph + RAG + Memory`
3. `青囊智诊 + 后续 agentic-RL`

### 6.2 不适合直接替代的地方

`HealthBench` 很强，但它不能完全替代我们的自建多轮 benchmark。

原因：

1. 我们要测的是 `agent 行为`
- 每轮问几个问题
- 高危提示是否前置
- 会不会把问题存进 `question_queue`
- 会不会自动生成首程病历

2. `HealthBench` 更偏向：
- 对最终回答做 rubric 评分

所以更合理的用法是：

- `HealthBench` 负责评估最终回答质量
- 我们自己的多轮 benchmark 负责评估 agent 过程

## 7. 对青囊智诊最推荐的接法

### 方案 A：最快落地

直接把 `HealthBench` 当成外部评测。

做法：

1. 用官方 `simple-evals`
2. 先跑支持的 API 模型，确认评测链路
3. 再写一个我们自己的 sampler，把 prompt 喂给 `青囊智诊`
4. 把最终输出交给官方 rubric grader

优点：

- 最快
- 最接近官方
- 面试里好讲

缺点：

- 需要适配一个 sampler
- 对本地多 agent 系统不是开箱即用

### 方案 B：更贴合当前项目

保留官方数据和 rubric 思路，但不强依赖官方 runner。

做法：

1. 读取 `HealthBench` 的 conversation + rubrics
2. 把最后一轮用户消息喂给 `青囊智诊`
3. 拿我们的最终回复去走 rubric grader
4. 输出：
- overall score
- safety 相关 rubric
- context seeking 相关 rubric
- communication 相关 rubric

优点：

- 更容易接入我们现有的 `LangGraph` 系统
- 更容易和我们自己的多轮 benchmark 放在一起

缺点：

- 不是完全原样调用官方命令

## 8. 对我们当前阶段的建议

当前最稳的顺序是：

1. 继续保留我们自己的多轮 benchmark
2. 把 `HealthBench` 加成一个“外部标准 benchmark”
3. 先做小规模抽样：
- `examples=30`
- `examples=50`
- `examples=100`
4. 等后续有了 `GRPO` 版本，再做前后对比

## 9. 我对这件事的判断

`HealthBench` 值得接，而且很适合面试时讲：

- 它足够新
- 它是医疗场景
- 它是多轮对话
- 它是 rubric 评测，不是很水的字符串匹配

但它不应该替代我们现在的多轮 agent benchmark。

更好的定位是：

- `HealthBench = 外部标准 benchmark`
- `青囊自建多轮 benchmark = agent 行为 benchmark`

两条线一起讲，项目会更完整。

## 10. 下一步怎么做

如果继续推进，我建议下一步做：

1. 写一个 `HealthBench sampler adapter`
2. 先抽 `30-50` 条样本跑通
3. 记录：
- overall
- safety
- context seeking
- communication
4. 和我们自己的多轮 benchmark 一起形成对照表
