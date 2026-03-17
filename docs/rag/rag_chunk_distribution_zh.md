# RAG Chunk 分布统计

## 总体规模

- 总 chunk 数：`1,840,456`
- 当前语料类型：全部来自中文 HF 医疗语料
- 当前已移除：`MSD 中文专业版`

## 按数据集统计

| 数据集 | chunk 数 | 占比 |
| --- | ---: | ---: |
| `FreedomIntelligence/huatuo_encyclopedia_qa` | 723,676 | 39.32% |
| `shibing624/medical` | 683,606 | 37.14% |
| `FreedomIntelligence/huatuo_knowledge_graph_qa` | 249,706 | 13.57% |
| `FreedomIntelligence/Huatuo26M-Lite` | 183,468 | 9.97% |

## 按 source_type 统计

| source_type | chunk 数 | 占比 |
| --- | ---: | ---: |
| `cn_medical_encyclopedia_qa` | 723,676 | 39.32% |
| `cn_medical_pretrain_text` | 683,606 | 37.14% |
| `cn_medical_kg_qa` | 249,706 | 13.57% |
| `cn_medical_high_quality_qa` | 183,468 | 9.97% |

## 结构解读

这份中文 RAG 库当前有两个很明显的特征：

1. 百科/教材型内容占比很高
- `huatuo_encyclopedia_qa + shibing624/medical` 合计约 `76.46%`
- 这意味着它非常适合做“疾病解释、检查解释、科普补充、长尾召回”

2. 高质量医疗 QA 占比还不算高
- `Huatuo26M-Lite` 只占约 `9.97%`
- 这说明当前库更像“宽覆盖知识底库”，而不是“高精度首程问诊底库”

3. 知识图谱 QA 适合作为补充层
- `huatuo_knowledge_graph_qa` 占约 `13.57%`
- 适合补结构化医学事实，但不适合单独承担患者友好解释

## 对项目的实际影响

### 对 `LiShiZhen-Education`
比较有帮助，因为：
- 科普解释
- 检查项目释义
- 疾病背景知识
- 患者友好说明

### 对 `BianQue-Intake`
帮助有限，因为：
- 首程问诊更依赖追问逻辑、信息缺失补问、分诊边界
- 这些能力主要还是要靠专门的首程 SFT 来补

### 对当前 demo 检索
当前总量已经足够大，但不适合直接用线性扫全文做在线检索。
因此更合理的做法是：

1. 保留这份 `184 万` 大库作为完整底库
2. 另做一份 `10万 - 30万 chunk` 的 demo 索引
3. 后面再把大库接入正式向量检索方案

## 当前结论

这 `184 万` chunk 已经足够支撑我们说“项目有一套大规模中文医疗 RAG 底库”，但它不应该直接决定文本 SFT 基模。
文本基模仍然应该优先考虑：
- 指令跟随
- 工具调用
- 多 Agent 调度
- 长期记忆融合
- 中文表达稳健性

这些维度上，当前更适合把 `Huatuo` 视为数据和视觉分支的重要资源，而不是直接替换整个文本主干。
