# RAG 文档源方案

## 1. 当前目标

这版 RAG 以“尽快做出可讲、可演示的医疗 Agent demo”为目标，不再追求过度复杂的来源混搭。
当前优先保留两类语料：

1. 中文 HF 医疗语料
2. 少量结构化官方资料（仅在英文补充场景使用）

用户已经明确放弃：
- MSD 中文专业版
- MedlinePlus Lab Tests

原因很简单：前者抓取质量不稳定，后者 HTML 结构不符合当前项目节奏。

## 2. 为什么当前主打 HF 中文医疗语料

因为我们现在最缺的不是“最权威的单页证据”，而是：

1. 足够宽的中文疾病覆盖面
2. 足够像真实问诊和医疗问答的中文表达
3. 能快速支持 `BianQue-Intake`、`LiShiZhen-Education` 和后续 RAG demo 的大规模语料

当前保留的中文主语料：

- `FreedomIntelligence/Huatuo26M-Lite`
- `FreedomIntelligence/huatuo_encyclopedia_qa`
- `FreedomIntelligence/huatuo_knowledge_graph_qa`
- `shibing624/medical`

这四类语料各自承担的作用：

- `Huatuo26M-Lite`：高质量中文医疗问答，适合做广覆盖召回
- `huatuo_encyclopedia_qa`：补疾病、检查、药物的百科问答层
- `huatuo_knowledge_graph_qa`：补结构化医学事实
- `shibing624/medical`：补教材、百科和通用中文医疗知识

## 3. RAG 流程里的 QR 和 HyDE

### QR

- 全称：`Query Rewriting`
- 作用：把用户口语化、信息不完整的表达改写成更适合检索的查询

示例：
- 用户原话：`我最近胸口堵得慌，还有点喘。`
- 改写后：`主诉=胸闷、气短；检索目标=红旗症状、常见鉴别、初步检查建议、何时急诊`

### HyDE

- 全称：`Hypothetical Document Embeddings`
- 作用：先生成一段理想化的“假设文档”，再用这段文档辅助检索更相关的语料

示例：
- `这是一名主诉胸闷和气短的患者，需要检索胸痛/呼吸困难相关危险信号、初步检查建议、门急诊分流和患者友好解释。`

## 4. CPU 阶段可以先做什么

在还没有切到 GPU 实例前，可以先完成：

1. 下载原始 HF 语料
2. 清洗 JSON / JSONL
3. 统一成结构化 record
4. 按 chunk 规则切分
5. 输出统一 JSONL 供后续 embedding 和建库

这些工作主要都是 CPU 和网络开销，不需要占用训练卡。

## 5. 当前脚本

- [download_rag_sources.py](../scripts/download_rag_sources.py)
- [build_rag_corpus.py](../scripts/build_rag_corpus.py)
- [cn_rag_hf_manifest.json](../configs/cn_rag_hf_manifest.json)

## 6. 推荐运行方式

先下载：

```bash
python scripts/download_rag_sources.py \
  --root /root/autodl-tmp/medagent/datasets/rag_raw \
  --only-cn \
  --with-cn-hf \
  --hf-manifest configs/cn_rag_hf_manifest.json \
  --hf-endpoint https://hf-mirror.com
```

再构建 chunk：

```bash
python scripts/build_rag_corpus.py \
  --raw-root /root/autodl-tmp/medagent/datasets/rag_raw \
  --only-cn \
  --out-file /root/autodl-tmp/medagent/rag/chunks/medical_corpus_cn.jsonl
```

## 7. 下一步增强点

1. 给不同来源加 `evidence_level`
2. 接入中文 embedding，例如 `bge-m3`
3. 增加 reranker
4. 把“医学知识索引”和“患者记忆索引”彻底拆开
5. 为 `LiShiZhen-Education` 增加引用式回答模板

## 8. 当前结论

对这个面试 demo 来说，先把 HF 中文医疗语料做大、做通、做稳定，比继续抓低质网页更重要。
