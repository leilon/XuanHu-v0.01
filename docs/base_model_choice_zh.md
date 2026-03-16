# 基模选择决策：是否直接采用 HuatuoGPT-Vision-7B

## 1. 结论先说

我建议采用折中方案：

- 文本主干继续使用 `Qwen/Qwen2.5-7B-Instruct`
- 视觉主干直接采用 `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

也就是说：

- 不把 `HuatuoGPT-Vision-7B` 当成整个系统唯一的统一基模
- 但把它当成视觉分支的起点

这是当前最省工作量、也最适合你这个项目展示方式的方案。

## 2. 为什么不建议让 HuatuoGPT-Vision 直接包打天下

因为它擅长的是“医学图文理解”，不是“C 端首程问诊总控”。

它的优势是：

- 已经在医学多模态上完成了一轮较强适配
- 基于 `Qwen2.5-VL` 架构，推理和部署都更顺
- 能直接减少我们在视觉对齐阶段的工作量

但它的已知局限也很明显：

- 训练核心还是 `PubMedVision`
- 更偏医学图像 / 图文问答
- 不是专门为互联网医疗首程问诊、多 Agent 调度、长期记忆、RAG 引用而训的
- 不等于已经适配“用户拍体检单、问化验单、问是否该门诊/急诊”的真实产品场景

所以更合理的做法是：

- 文本主干负责问诊、分诊、用药、记忆、工具调用
- 视觉主干负责图片理解、报告抽取、异常项解释

## 3. 采用 HuatuoGPT-Vision-7B 以后，我们省掉了什么

主要省掉的是“从零做视觉对齐”的这一步。

如果我们直接采用：

- `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

那么我们可以：

- 跳过最重的视觉对齐预训练
- 不必自己先跑完整 `PubMedVision` 注入流程
- 直接做“面向我们场景的视觉 SFT”

这会明显减少前期工作量和 GPU 成本。

## 4. 采用 HuatuoGPT-Vision-7B 后，仍然必须补什么

即便采用它，我们仍然要补一轮“项目场景 SFT”。

### 必补的数据方向

#### 4.1 检验报告 / 化验单解读

重点覆盖：

- 血常规
- 肝肾功能
- 血脂
- 血糖 / 糖化
- 尿常规
- 甲功

目标输出不是纯 OCR，而是：

- 异常项摘要
- 和当前症状的关联提示
- 下一步建议
- 是否需要门诊 / 急诊

#### 4.2 患者友好解释

`HuatuoGPT-Vision` 的医学表达可能偏专业，我们要补：

- 面向普通用户的解释
- 避免过度诊断
- 风险提示分层
- 明确“不能替代线下医生”的边界

#### 4.3 报告图片质量差的场景

用户实际上传的图片通常会有：

- 倾斜
- 模糊
- 反光
- 局部截断
- 手机拍照背景噪声

这类数据 `PubMedVision` 不能完全覆盖，需要我们自己补。

## 5. 如果采用 HuatuoGPT-Vision-7B，我们要准备哪些训练文件

采用后，视觉分支建议只做“任务化 SFT”。

### 5.1 视觉 SFT 主文件

建议统一成 JSONL，每条一行。

推荐字段：

```json
{
  "image": "images/report_0001.jpg",
  "prompt": "请先提取这份化验单中的关键异常项，再用患者能听懂的话解释可能意义，并说明下一步建议。",
  "response": "这份化验单里最需要关注的是白细胞升高和C反应蛋白升高，提示体内可能存在感染或炎症。结合你现在发热和咳嗽的症状，建议尽快到呼吸科或发热门诊进一步评估。"
}
```

### 5.2 视觉 SFT 子桶建议

建议拆成四个文件桶：

- `vision_report_extract.jsonl`
  - 只训练抽取关键指标与异常项
- `vision_report_explain.jsonl`
  - 训练把专业指标翻译成患者能理解的话
- `vision_report_triage.jsonl`
  - 训练结合症状给出下一步就诊建议
- `vision_report_followup.jsonl`
  - 训练结合长期记忆和既往病史做复诊判断

### 5.3 视觉数据的清洗规则

- 去掉纯学术图像但与我们场景差距太大的样本
- 去掉没有明确问答目标的图文对
- 去掉只给结论、不解释依据的样本
- 去掉明显过度诊断、断言式回答
- 统一输出风格为“摘要 + 风险提示 + 下一步建议”

## 6. 如果不采用 HuatuoGPT-Vision-7B，我们需要补什么

如果不用它，而是自己从 `Qwen/Qwen2.5-VL-7B-Instruct` 起步，那么就要补完整两阶段训练。

### 阶段 A：视觉对齐 / 视觉预训练

最关键的数据：

- `FreedomIntelligence/PubMedVision`

官方仓库已经明确说：

- 复现实验建议使用 `PubMedVision + LLaVA dataset`
- Qwen2.5-VL 路线分成：
  - 初始化 VL
  - Vision Alignment
  - Vision Instruction Fine-tuning

### 阶段 B：医疗视觉指令微调

除了 `PubMedVision`，还建议再补：

- `VQA-RAD`
- `SLAKE`
- `PathVQA`
- `PMC-VQA`
- 自建报告/化验单解释数据

注意：

- 这些公开 benchmark 只能用训练集或公开训练划分
- 不能直接把测试集拿去训练

## 7. 文本主干要不要也换成 Huatuo 系列

我的建议是暂时不要。

原因：

- `HuatuoGPT-II-7B` 的 7B 公开底座不是 Qwen，而是 `Baichuan2-7B-Base`
- 我们现在文本和视觉都在围绕 `Qwen2.5` 生态设计
- 如果文本换成 Huatuo-II，视觉换成 Qwen2.5-VL / HuatuoVision，会让训练和部署链路变复杂

所以当前最稳的是：

- 文本：`Qwen2.5-7B-Instruct`
- 视觉：`HuatuoGPT-Vision-7B-Qwen2.5VL`

## 8. 对我们项目的最终建议

### 推荐方案

- 文本分支：
  - `Qwen/Qwen2.5-7B-Instruct`
  - 做首程问诊、多 Agent、RAG、长期记忆、工具调用的分桶 SFT
- 视觉分支：
  - `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`
  - 直接做报告/检验单/患者友好解释的二次 SFT

### 不推荐方案

- 让 `HuatuoGPT-Vision-7B` 直接兼任全部文本主流程
- 自己从零重跑完整视觉对齐，除非后面你决定把多模态做成项目绝对主线

## 9. 这会如何影响我们接下来的脚本

如果采用推荐方案：

- 文本训练脚本保留“第一阶段文本 SFT”
- 视觉训练脚本改成“基于 HuatuoGPT-Vision 的任务化 SFT”
- 不再把视觉第一阶段默认设成必须重跑的 alignment pretrain

如果后续你坚持从 `Qwen2.5-VL` 原始底模起步：

- 我们再保留“视觉对齐脚本 + 视觉 SFT 脚本”双阶段

## 10. 参考来源

- [HuatuoGPT-Vision 官方仓库](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)
- [HuatuoGPT-Vision Qwen2.5VL 模型卡](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL)
- [HuatuoGPT-II 官方仓库](https://github.com/FreedomIntelligence/HuatuoGPT-II)
