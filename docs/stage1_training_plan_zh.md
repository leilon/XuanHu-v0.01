# 第一阶段训练方案：文本与视觉 SFT

## 1. 先说结论

对于你当前这套资源和目标，我建议第一阶段这样做：

- 文本：不做继续预训练，直接做分桶 SFT
- 视觉：如果采用 `HuatuoGPT-Vision-7B-Qwen2.5VL`，直接做任务化 SFT

## 2. 为什么文本第一阶段不建议做继续预训练

### 2.1 不是因为继续预训练没用

而是因为对我们当前项目来说，性价比不高。

`Qwen2.5-7B-Instruct` 已经有很强的：

- 中文表达能力
- 指令跟随能力
- 基础工具调用能力

我们现在真正缺的不是“更多通用医学词汇”，而是：

- 医生式首程问诊流程
- 分诊边界
- 长期记忆使用
- RAG 证据引用
- 多 Agent 协作

这些更像是 `SFT` 要补的能力，而不是继续预训练最擅长补的能力。

### 2.2 对 4 卡 5090 更现实

继续预训练的特点是：

- 吃数据量更大
- 训练时间更长
- 对语料质量和去重要求更高
- 很容易把钱花在“提升不明显”的地方

而第一阶段分桶 SFT 的特点是：

- 目标更明确
- 训练周期更短
- 更容易在面试里讲清楚“我到底补了什么能力”

## 3. 文本第一阶段应该怎么训

建议分成四个桶混合训练：

1. 通用中文医疗问答
2. 首程问诊与结构化病史采集
3. 分诊与安全边界
4. RAG / 工具调用 / 长期记忆融合

当前脚本：

- [prepare_text_sft_stage1.py](../scripts/prepare_text_sft_stage1.py)
- [train_text_sft_stage1.py](../scripts/train_text_sft_stage1.py)
- [text_sft_stage1.sample.jsonl](../data/text_sft_stage1.sample.jsonl)

注意：

- `text_sft_stage1.sample.jsonl` 只是我手写的骨架示例，不是直接从公开数据集原样抽出来的
- 第一阶段 `BianQue` 文本 SFT 不应该吃长篇解释答案
- `prepare_text_sft_stage1.py` 现在会默认把长篇说明式回答压成“首轮追问式短回答”
- 长解释型医疗 QA 更适合放到后续“患者解释 / 科普 / 报告说明”类 adapter

推荐先生成训练文件：

```bash
python scripts/prepare_text_sft_stage1.py \
  --sft-root /root/autodl-tmp/medagent/datasets/sft \
  --train-out /root/autodl-tmp/medagent/datasets/sft/train_stage1_text.jsonl \
  --valid-out /root/autodl-tmp/medagent/datasets/sft/valid_stage1_text.jsonl
```

推荐启动方式：

```bash
accelerate launch --config_file configs/accelerate_4x5090.yaml scripts/train_text_sft_stage1.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --train-file /root/autodl-tmp/medagent/datasets/sft/train_stage1_text.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/sft/valid_stage1_text.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/bianque_text_stage1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project qingnang-clinicos \
  --wandb-run-name text-stage1-bianque
```

## 4. 为什么单个多模态模型训练和推理成本更高

这点很关键。

“一个模型统一做文本和视觉”看起来更简洁，但成本通常更高，主要有四个原因。

### 4.1 模型本体更重

`VL` 模型除了语言部分，还带：

- 视觉编码器 / 视觉塔
- 多模态投影层
- 图像 token 处理逻辑

所以即使都是“7B”级别，`VL` 也不等于和纯文本 7B 一样轻。

### 4.2 文本场景也要背着视觉模块

你这个项目里绝大多数请求还是文本：

- 首程问诊
- 分诊
- RAG
- 长期记忆
- 问药

如果全都走同一个 `VL` 模型，那么哪怕只是纯文本请求，你也在运行一个更重的系统。

### 4.3 视觉样本会拉长序列和激活显存

图像会被编码成视觉 token。

这意味着：

- 单条样本更长
- 激活显存更大
- batch size 更小
- 梯度累积更重

所以同样是 `QLoRA`，视觉 SFT 通常会更挤显存。

### 4.4 一个模型同时学文本主流程和视觉主流程，更容易“抢容量”

如果一个模型既负责：

- 文本问诊
- 工具调用
- 长期记忆
- 报告解读
- 图文推理

那么它在有限参数空间里要同时适配更多任务，训练目标也更容易互相拉扯。

## 5. 为什么视觉第一阶段建议直接基于 HuatuoVision 做任务化 SFT

因为它已经替我们完成了“从通用 Qwen2.5-VL 到医学视觉模型”的最重一段。

所以如果采用：

- `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

我们可以直接把训练目标聚焦到：

- 报告异常抽取
- 患者友好解释
- 结合症状给下一步建议
- 结合既往史做复诊判断

当前脚本：

- [validate_vision_sft_data.py](../scripts/validate_vision_sft_data.py)
- [train_vision_sft_stage1.py](../scripts/train_vision_sft_stage1.py)
- [vision_report_sft.sample.jsonl](../data/vision_report_sft.sample.jsonl)

正式训练前建议先校验标注文件：

```bash
python scripts/validate_vision_sft_data.py \
  --file /root/autodl-tmp/medagent/datasets/sft/vision_report_train.jsonl
```

推荐启动方式：

```bash
accelerate launch --config_file configs/accelerate_4x5090.yaml scripts/train_vision_sft_stage1.py \
  --base-model /root/autodl-tmp/medagent/models/huatuogpt-vision-7b-qwen2.5vl \
  --train-file /root/autodl-tmp/medagent/datasets/sft/vision_report_train.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/sft/vision_report_valid.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/canggong_vision_stage1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project qingnang-clinicos \
  --wandb-run-name vision-stage1-canggong
```

## 6. 如果后面你坚持统一成一个模型

也不是不行。

那更合理的顺序是：

1. 先用两模型把系统跑通
2. 再用统一多模态模型做蒸馏或收敛

这样更稳，而不是一开始就把所有复杂度压到一个模型上。
