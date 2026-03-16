# RAG 文档源与 CPU 预处理方案

## 1. 目标

这一版 RAG 不追求“海量文档”，而追求和诊断学、首程问诊、初步检查、用药安全密切相关。

## 2. 当前优先文档源

### 2.1 MedlinePlus Health Topics

用途：

- 疾病与症状的患者友好解释
- 常见表现
- 就医建议的通俗表达

优点：

- 官方来源
- 结构化 XML
- 适合先做健康教育与基础问答层

### 2.2 MedlinePlus Lab Tests

用途：

- 初步检查建议
- 检验项目的解释
- 报告解读时的背景知识

优点：

- 官方来源
- 和“诊断学里的化验检查”高度相关
- 非常适合你的报告/检验单场景

### 2.3 openFDA Drug Labels

用途：

- 适应证
- 禁忌证
- 警示
- 药物相互作用

优点：

- 结构化 JSON
- 和问药、禁忌、安全边界直接相关

## 3. 为什么这三类最适合第一版

因为它们刚好对应你项目最需要的三条知识链：

1. 症状 / 疾病解释
2. 初步检查 / 报告解读
3. 用药安全

## 4. RAG 流程里的 QR 和 HyDE

### QR

- 全称：`Query Rewriting`
- 作用：把用户口语化、不完整的问题重写成更适合检索的查询

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

1. 下载文档
2. 清洗 HTML / XML / JSON
3. 结构化抽取
4. chunk 切分
5. 保存成统一 JSONL

这些都是 `CPU` 活，完全可以先做。

## 6. 已写好的脚本

- [download_rag_sources.py](../scripts/download_rag_sources.py)
- [build_rag_corpus.py](../scripts/build_rag_corpus.py)

推荐先跑：

```bash
python scripts/download_rag_sources.py --root /root/autodl-tmp/medagent/datasets/rag_raw
python scripts/build_rag_corpus.py \
  --raw-root /root/autodl-tmp/medagent/datasets/rag_raw \
  --out-file /root/autodl-tmp/medagent/rag/chunks/medical_corpus.jsonl
```

## 7. 参考来源

- [MedlinePlus XML](https://medlineplus.gov/xml.html)
- [MedlinePlus Lab Tests](https://medlineplus.gov/lab-tests/)
- [openFDA Drug Label API](https://open.fda.gov/apis/drug/label/)
