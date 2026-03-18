# 多轮 API Benchmark 运行说明

## 1. 目标

这套 benchmark 用来观察 `青囊智诊` 在多轮问诊状态下的真实行为，而不是只看单轮问答。

它关注三件事：

1. 模型是否会在高危场景下优先提示就医
2. 模型是否能把每轮追问控制在 1 到 2 个问题
3. 模型是否能在 5 到 10 轮内收束到初步判断、分诊建议和检查建议

## 2. 当前实现

- 调度框架：`LangGraph`
- 多轮入口：`LangChainOrchestrator.run_visit_turn()`
- 病人模拟器：`medagent/benchmark/patient_simulator.py`
- 运行脚本：`scripts/run_patient_sim_benchmark.py`
- 场景文件：`data/sim_patient_cases.json`

## 3. 数据结构

每条 case 至少包含：

- `id`
- `name`
- `age`
- `sex`
- `opening`
- `hidden_case`
- `expected`

其中：

- `opening` 是用户第一句
- `hidden_case` 是给病人模拟器和后续 judge 用的隐藏病例设定
- `expected` 是当前简化版 evaluator 的关键词参考

## 4. 运行方式

```bash
python scripts/run_patient_sim_benchmark.py \
  --scenarios data/sim_patient_cases.json \
  --max-turns 6 \
  --report-file docs/reports/multiturn_api_benchmark_report.json
```

## 5. 下一步

后续可以把这条 benchmark 继续升级为：

1. 强 LLM 病人模拟器
2. 强 LLM judge
3. 更严格的多轮行为指标
4. agentic-RL 前后效果对比基准
