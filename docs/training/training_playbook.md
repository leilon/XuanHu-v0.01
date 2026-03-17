# 训练方案总览

这份文档只保留当前真正执行的训练路线，不再混用旧方案。

开始看训练前，建议先读：
- [SFT / Agent / Adapter 定位](./sft_adapter_strategy_zh.md)
- [BianQue-Intake SFT 说明](./bianque_intake_sft_zh.md)
- [第一阶段训练方案](./stage1_training_plan_zh.md)

## 1. 基本决策

- 文本基座：`Qwen2.5-7B-Instruct`
- 视觉基座：`HuatuoGPT-Vision-7B-Qwen2.5VL`
- 第一阶段先做 `SFT`
- `Agentic-RL` 放在最后

原因很明确：
- 现在最缺的是首程问诊、解释风格、报告理解这些任务能力
- 不是继续预训练通用医学语料
- 也不是过早把所有 agent 都拆成独立 adapter

## 2. AutoDL 数据路径

统一工作根目录：
- `/root/autodl-tmp/medagent`

原始下载数据：
- `/root/autodl-tmp/medagent/datasets/sft`
- `/root/autodl-tmp/medagent/datasets/rl`
- `/root/autodl-tmp/medagent/datasets/eval`

任务化清洗数据：
- `/root/autodl-tmp/medagent/datasets/curated/bianque_intake`
- `/root/autodl-tmp/medagent/datasets/curated/text_stage1`
- `/root/autodl-tmp/medagent/datasets/curated/lishizhen_education`
- `/root/autodl-tmp/medagent/datasets/curated/canggong_vision`

训练产物：
- `/root/autodl-tmp/medagent/outputs/adapters`

## 3. 目录分工

- `trainer/data/`
  - 下载训练数据
  - 清洗原始数据

- `trainer/text/`
  - 文本 SFT 数据准备
  - 文本 adapter 训练

- `trainer/vision/`
  - 视觉数据校验
  - 视觉 adapter 训练

- `trainer/rl/`
  - 偏好数据构建
  - DPO / Agentic-RL 训练

- `trainer/core/`
  - 通用 QLoRA 入口

## 4. 当前推荐训练顺序

1. 下载原始数据到 `datasets/sft`
2. 清洗出 `BianQue-Intake` 数据到 `datasets/curated/bianque_intake`
3. 训练 `BianQue-Intake` adapter
4. 清洗出 `LiShiZhen-Education` 数据
5. 训练 `LiShiZhen-Education` adapter
6. 准备视觉报告数据
7. 训练 `CangGong-Report` adapter
8. 补 benchmark
9. 最后再做 `Agentic-RL`

## 5. 当前脚本入口

下载与统一清洗：

```bash
python trainer/data/download_medical_datasets.py --root /root/autodl-tmp/medagent/datasets
python trainer/data/prepare_medical_training_data.py --root /root/autodl-tmp/medagent/datasets
```

BianQue 数据准备：

```bash
python trainer/text/prepare_bianque_intake_sft.py \
  --sft-root /root/autodl-tmp/medagent/datasets/sft \
  --train-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/train.jsonl \
  --valid-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/valid.jsonl \
  --summary-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/summary.json
```

BianQue 训练：

```bash
accelerate launch --config_file configs/accelerate_4x5090.yaml trainer/text/train_bianque_intake_sft.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --train-file /root/autodl-tmp/medagent/datasets/curated/bianque_intake/train.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/curated/bianque_intake/valid.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/bianque_intake_stage1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project qingnang-clinicos \
  --wandb-run-name bianque-intake-stage1
```

视觉训练：

```bash
python trainer/vision/validate_vision_sft_data.py \
  --file /root/autodl-tmp/medagent/datasets/sft/vision_report_train.jsonl

accelerate launch --config_file configs/accelerate_4x5090.yaml trainer/vision/train_vision_sft_stage1.py \
  --base-model /root/autodl-tmp/medagent/models/huatuogpt-vision-7b-qwen2.5vl \
  --train-file /root/autodl-tmp/medagent/datasets/sft/vision_report_train.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/sft/vision_report_valid.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/canggong_vision_stage1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project qingnang-clinicos \
  --wandb-run-name vision-stage1-canggong
```

## 6. QLoRA 在这个项目里的定位

`QLoRA` 不是为了做 test-time training。

当前最合适的用法是：
- 低成本 SFT
- 周期性持续学习
- teacher 蒸馏后的任务适配
- 多任务 adapter 管理

## 7. Agentic-RL 的定位

当前先不作为主线。

原因：
- 首程问诊数据还在整理
- `LiShiZhen-Education` 还没单独清洗
- 视觉报告链路还没稳定
- RAG 和长期记忆仍以系统设计为主

等 SFT、RAG、benchmark 稳住之后，再用 `trainer/rl/` 下的脚本推进偏好优化。
