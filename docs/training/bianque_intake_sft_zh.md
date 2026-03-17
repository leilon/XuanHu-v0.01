# BianQue-Intake SFT 说明

## 1. 这条训练线要解决什么问题

`BianQue-Intake` 不是做长篇科普的，它负责的是：

1. 把用户口语化、信息不完整的主诉接住
2. 在第一轮补齐关键病史
3. 识别红旗症状
4. 为后续分诊、RAG、工具调用打基础

所以它的训练目标不是“回答得很全”，而是“追问得很对”。

## 2. 数据风格

这条 SFT 只保留短轮次首程问诊风格：

- 输入：用户主诉
- 输出：首轮追问 + 必要的紧急就医提醒

不做：
- 长篇疾病科普
- 大段药物说明
- 过早诊断结论

## 3. 数据准备脚本

脚本：
- [prepare_bianque_intake_sft.py](../../trainer/text/prepare_bianque_intake_sft.py)

它会做几件事：

1. 从混合中文医疗数据里抽取用户问题
2. 过滤掉纯科普、纯定义类问题
3. 把症状主诉映射到首程问诊模板
4. 自动补充与病例类型相关的追问
5. 生成适合 BianQue 的短轮次 SFT 数据

## 4. 训练脚本

脚本：
- [train_bianque_intake_sft.py](../../trainer/text/train_bianque_intake_sft.py)

训练时会固定一个系统角色：
- 你是 `BianQue-Intake`
- 先追问、先识别风险
- 不要一上来给长解释

## 5. 典型命令

先准备数据：

```bash
python trainer/text/prepare_bianque_intake_sft.py \
  --sft-root /root/autodl-tmp/medagent/datasets/sft \
  --train-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/train.jsonl \
  --valid-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/valid.jsonl \
  --summary-out /root/autodl-tmp/medagent/datasets/curated/bianque_intake/summary.json
```

再训练：

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

## 5.1 数据位置约定

- 原始下载数据：`/root/autodl-tmp/medagent/datasets/sft`
- 清洗后的 BianQue 数据：`/root/autodl-tmp/medagent/datasets/curated/bianque_intake`

这样做的目的是把“原始语料”和“首程问诊任务数据”拆开，避免后面训练时再把未清洗语料直接喂给 adapter。

## 6. 和 LiShiZhen-Education 的边界

`BianQue-Intake` 负责：
- 追问
- 补病史
- 识别风险
- 首轮分流

`LiShiZhen-Education` 负责：
- 科普解释
- 报告释义
- 患者友好说明
- 诊后教育

这两个 adapter 不应该混训。

