# 训练方案总览（AutoDL + 7B）

## 1. 基模选择

### 文本模型

- `Qwen/Qwen2.5-7B-Instruct`

### 视觉模型

- `Qwen/Qwen2.5-VL-7B-Instruct`

选择原因：

- 中文能力较强
- 指令跟随和工具调用表现稳
- QLoRA 生态成熟
- 比较适合你这个“医疗 Agent + RAG + 多轮问诊”的项目

## 2. AutoDL 数据盘布局

统一使用：

- `/root/autodl-tmp/medagent`

建议子目录：

- `models/`：基模
- `datasets/sft/`：SFT 数据
- `datasets/rl/`：Agentic-RL 数据
- `datasets/rag_raw/`：RAG 原始文档
- `rag/`：切分结果和索引
- `outputs/adapters/`：LoRA adapter
- `wandb/`：W&B 日志
- `hf_cache/`：Hugging Face 缓存

初始化命令：

```bash
bash scripts/autodl_prepare_workspace.sh /root/autodl-tmp/medagent
```

## 3. SFT 数据选择

推荐主料：

- `BillGPT/Chinese-medical-dialogue-data`
- `wangrongsheng/cMedQA-V2.0`
- `FreedomIntelligence/Medical-R1-Distill-Data-Chinese`
- `FreedomIntelligence/HuatuoGPT-sft-data-v1`
- `shibing624/medical` 中过滤后的高质量中文子集

还要补内部数据：

- 首程问诊轨迹
- 分诊结论
- 报告解释
- 问药和药物禁忌
- 结合长期记忆的复诊场景

最终统一成：

```json
{"input":"...", "output":"..."}
```

## 4. QLoRA 阶段

先做 SFT adapter 训练：

```bash
export WANDB_API_KEY=xxx
bash scripts/run_sft_with_wandb.sh /root/autodl-tmp/medagent
```

或直接运行：

```bash
python3 scripts/train_qlora.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --train-file /root/autodl-tmp/medagent/datasets/sft/train_v1.jsonl \
  --replay-file /root/autodl-tmp/medagent/datasets/sft/replay_v0.jsonl \
  --task general_intake \
  --dataset-name med_sft_v1 \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/general_intake_v1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project medagent-7b \
  --wandb-run-name sft-general-intake-v1 \
  --wandb-dir /root/autodl-tmp/medagent/wandb
```

## 5. Agentic-RL 数据策略

公开可直接用的数据只适合作种子，不足以覆盖真实问诊行为。

更合理的组成是：

- 小规模公开 medical preference / reward 数据
- 你的多 Agent 系统真实日志
- 基于 benchmark 和病例包合成的 agentic preference pairs

推荐第一版配比：

- `60%` 合成 agentic preference
- `20%` 多 Agent 真实日志回流
- `10%` 公开 medical preference 数据
- `10%` 高风险人工构造负样本

重点 failure mode：

- 漏问过敏史
- 漏问慢病和长期用药
- 男性误问怀孕
- 育龄女性该问妊娠却没问
- 危险信号没有升级急诊
- 明明需要报告/药物工具却不调用

## 6. Agentic-RL 训练阶段

第一步建议先做 DPO/ORPO 一类的偏好优化，而不是一上来就做复杂在线 RL。

示例：

```bash
python3 scripts/train_agentic_rl.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --pairs-file /root/autodl-tmp/medagent/datasets/rl/agentic_pairs_v1.jsonl \
  --task agentic_policy \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/agentic_policy_v1 \
  --adapter-bank-dir /root/autodl-tmp/medagent/outputs/adapters \
  --cache-dir /root/autodl-tmp/medagent/hf_cache \
  --wandb-project medagent-7b \
  --wandb-run-name rl-agentic-policy-v1 \
  --wandb-dir /root/autodl-tmp/medagent/wandb
```

## 7. 多模态阶段

视觉分支重点先做：

- 检验报告
- 化验单
- 结构化图文问答

不要一上来就做重影像诊断。

训练数据建议先聚焦：

- 报告图片 + 问题
- OCR 文本 + 问题
- 输出异常摘要、风险提示和下一步建议

## 8. 最终评测

跑离线 benchmark：

```bash
python3 -m medagent.benchmark.run --dataset data/benchmark_cases.json
```

再跑多轮病人模拟 benchmark：

```bash
python scripts/run_patient_sim_benchmark.py --scenarios data/sim_patient_cases.json
```

## 9. W&B 建议跟踪指标

- `loss`
- `learning_rate`
- `grad_norm`
- 高风险症状召回率
- grounding 引用率
- 工具调用成功率
- benchmark 总分

## 10. 建议和哪些文档联动看

- `docs/architecture.md`
- `docs/agentic_rl_synthesis_zh.md`
- `docs/agentic_rl_dataset_survey_zh.md`
- `docs/clinical_first_visit_prompt_zh.md`
- `docs/benchmark_plan.md`

## 补充文档

- [Huatuo 系列模型调研](./huatuo_series_research_zh.md)
- [基模选择决策](./base_model_choice_zh.md)
