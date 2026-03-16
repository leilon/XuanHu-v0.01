# RAG 文档源与 CPU 预处理方案

## 1. 这次的目标

这一版 RAG 不再只做“小而精”的 demo 语料，而是升级成两层结构：

1. 强证据层
   - 临床参考手册
   - 检验检查说明
   - 药品标签
   - 指南页
2. 大体量中文补充层
   - 中文医学百科
   - 中文医患问答
   - 中文医疗教材片段
   - 中文医疗对话

这样做的原因很直接：

- 首程问诊和分诊，需要足够宽的疾病覆盖面
- 报告解读和问药，需要强证据资料兜底
- 科普解释和长尾病种，需要大语料补充召回

## 2. 语料规模应该做到什么量级

如果是为了做一个真正能支撑 `medicine agent` 的系统，我建议：

- 原始语料：至少 `1GB+`
- 更舒服的目标：`2GB - 8GB`
- 清洗后 chunk 语料：通常会比原始网页更小，但依然可能有数十万到上百万 chunk

所以你说“至少得上 GB”是对的。  
真正重要的不是单纯堆体量，而是把不同证据等级分层：

- `clinical_reference_cn / drug_label / medical_test`：高权重
- `cn_medical_encyclopedia_qa / cn_medical_dialogue`：中低权重，作为召回补充

## 3. 当前采用的两条主线

### 3.1 官方与结构化资料

这一层继续保留：

- `MedlinePlus Health Topics`
- `MedlinePlus Lab Tests`
- `openFDA Drug Labels`

用途：

- 症状与疾病解释
- 初步检查和检验指标解释
- 用药禁忌和药物安全

### 3.2 中文大体量资料

这一层新增两类：

#### A. Hugging Face / hf-mirror 中文医疗库

当前准备接入：

- `FreedomIntelligence/huatuo_encyclopedia_qa`
- `FreedomIntelligence/huatuo_knowledge_graph_qa`
- `FreedomIntelligence/Huatuo26M-Lite`
- `shibing624/medical`
- `wangrongsheng/cMedQA-V2.0`
- `ticoAg/Chinese-medical-dialogue`

这些源的意义不一样：

- `huatuo_encyclopedia_qa`：覆盖广，适合补疾病和药物知识
- `huatuo_knowledge_graph_qa`：适合补结构化医学事实
- `Huatuo26M-Lite`：高质量中文医疗 QA，可做宽覆盖补充
- `shibing624/medical`：里面的 `medical_book_zh` 和 `train_encyclopedia` 很适合做教材/百科层
- `cMedQA-V2.0`：医患问答补充
- `Chinese-medical-dialogue`：真实问诊口语表达补充

#### B. 中文临床参考手册

这次新增：

- `默沙东诊疗手册中文专业版（MSD Manuals CN Professional）`

优点：

- 中文专业内容
- 按学科组织
- 覆盖我们关心的：
  - 内科学
  - 外科学
  - 诊断相关专题
  - 妇产科学
  - 儿科学
  - 神经病学

脚本层面会通过 sitemap 过滤这些学科页面，再做本地 html 抽取和 chunk。

## 4. RAG 流程里的 QR 和 HyDE

### QR

- 全称：`Query Rewriting`
- 作用：把用户口语化、不完整的问题改写成更适合检索的查询

例如：

- 用户原话：`我最近总觉得很难受，胸口有点堵，还喘。`
- 重写后：`主诉=胸闷、气短；检索目标=危险信号、常见鉴别方向、首批检查建议、急诊指征`

### HyDE

- 全称：`Hypothetical Document Embeddings`
- 作用：先生成一段“假设性的理想文档”，再用它帮助检索更相关内容

例如：

- `这是一名主诉胸闷和气短的患者，需重点检索胸痛/呼吸困难相关红旗症状、急诊指征、初步检查建议和患者友好解释。`

## 5. CPU 阶段先做什么

在没租 GPU 的时候，可以先做：

1. 下载原始文档和数据集
2. 清洗 HTML / XML / JSON / JSONL
3. 统一成结构化 record
4. chunk 切分
5. 保存成统一 JSONL

这些主要都是 `CPU` 活。

更具体一点：

- `下载 / 清洗 / chunk`：CPU 为主
- `embedding / reranker / 向量建库`：CPU 也能做，GPU 更快

所以现在完全可以先把 chunk 准备好，再去租卡做向量化和训练。

## 6. 这次新增的脚本和配置

- [download_rag_sources.py](../scripts/download_rag_sources.py)
- [build_rag_corpus.py](../scripts/build_rag_corpus.py)
- [cn_rag_hf_manifest.json](../configs/cn_rag_hf_manifest.json)

## 7. 推荐运行方式

先下载：

```bash
python scripts/download_rag_sources.py \
  --root /root/autodl-tmp/medagent/datasets/rag_raw \
  --with-cn-hf \
  --with-msd-cn \
  --hf-manifest configs/cn_rag_hf_manifest.json \
  --hf-endpoint https://hf-mirror.com \
  --msd-max-pages 1200
```

再构建 chunk：

```bash
python scripts/build_rag_corpus.py \
  --raw-root /root/autodl-tmp/medagent/datasets/rag_raw \
  --out-file /root/autodl-tmp/medagent/rag/chunks/medical_corpus.jsonl
```

## 8. 后续要继续增强的点

1. 给不同来源加 `evidence_level`
2. 增加中文指南 PDF / 网页解析器
3. 做中文 `bge-m3` embedding
4. 加 reranker
5. 医学知识索引和患者记忆索引彻底拆开

## 9. 参考来源

- [MedlinePlus XML](https://medlineplus.gov/xml.html)
- [MedlinePlus Lab Tests](https://medlineplus.gov/lab-tests/)
- [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/)
- [FreedomIntelligence/huatuo_encyclopedia_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa)
- [FreedomIntelligence/huatuo_knowledge_graph_qa](https://huggingface.co/datasets/FreedomIntelligence/huatuo_knowledge_graph_qa)
- [FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)
- [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- [wangrongsheng/cMedQA-V2.0](https://huggingface.co/datasets/wangrongsheng/cMedQA-V2.0)
- [ticoAg/Chinese-medical-dialogue](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue)
- [hf-mirror](https://hf-mirror.com)
- [MSD 手册中文专业版](https://www.msdmanuals.cn/professional)
