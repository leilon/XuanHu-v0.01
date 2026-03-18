# HealthBench 开跑前检查清单

## 1. 目录准备

建议把官方仓库放在：

`/root/autodl-tmp/medagent/benchmarks/simple-evals`

要求该目录下存在：

- `simple_evals.py`
- `healthbench_eval.py`

## 2. 环境准备

至少需要：

- Python 可运行
- `OPENAI_API_KEY` 已设置
- `simple-evals` 依赖已按官方方式安装

## 3. 我们项目里的调用脚本

已准备：

- [run_healthbench_official.py](C:\Users\leimi\Desktop\agentic_medical_gpt\scripts\run_healthbench_official.py)

## 4. 推荐第一轮命令

```bash
python scripts/run_healthbench_official.py \
  --simple-evals-dir /root/autodl-tmp/medagent/benchmarks/simple-evals \
  --eval healthbench \
  --model gpt-4.1 \
  --examples 30 \
  --n-threads 16
```

## 5. 建议先跑的顺序

1. `healthbench`
2. `healthbench_consensus`
3. `healthbench_hard`

## 6. 推荐先记录的指标

- 总分
- 运行样本数
- 使用模型
- 运行时间
- 是否为多轮/最终回答模式
