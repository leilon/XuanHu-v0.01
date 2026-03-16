# 医疗 Agent 的 RAG 数据库设计

## 1. 知识源分层

RAG 不应该只是一锅混合文档，而应该按医疗任务拆层：

### 临床指南

用于：

- 分诊建议
- 首程问诊建议
- 初步检查建议
- 危险信号识别

### 药品知识

用于：

- 适应证
- 禁忌证
- 药物相互作用
- 剂量范围提示

### 面向患者的科普内容

用于：

- 把专业内容翻译成用户可理解的话
- 降低回答的阅读门槛

### 本地服务流程

用于：

- 医院科室对应关系
- 线上 / 线下路由
- 是否建议门诊或急诊

## 2. 存储布局

建议路径：

- 原始文档：`/root/autodl-tmp/medagent/datasets/rag_raw`
- 切分后的 chunk：`/root/autodl-tmp/medagent/rag/chunks`
- 向量索引：`/root/autodl-tmp/medagent/rag/index`

每个 chunk 至少带这些 metadata：

- `source_id`
- `source_type`
  - `guideline`
  - `drug`
  - `education`
  - `process`
- `disease_tags`
- `drug_tags`
- `updated_at`
- `evidence_level`

## 3. 检索流程

推荐采用分阶段检索：

1. 先做 query 分类
   - 分诊
   - 问药
   - 报告解读
2. 再做混合检索
   - BM25 稀疏检索
   - dense embedding 检索
3. 对 top-K 结果做 rerank
4. 最终生成时要求带证据引用

## 4. 向量模型与索引

原型阶段推荐：

- embedding 模型：`BAAI/bge-m3`
- 向量库：
  - `FAISS`：适合原型
  - `Milvus`：适合后续扩展

## 5. 质量与安全要求

RAG 在医疗场景里必须额外看两件事：

### grounding

- 回答里是否真的引用了知识库
- 引用的内容是否和最终建议一致

### 安全性

- 有没有漏掉危险信号
- 有没有漏掉禁忌证
- 有没有在不该保守时仍然让用户居家观察

## 6. 评测指标

建议至少跟踪：

- 引用命中率
- 医学正确性
- 高风险症状召回率
- 平均检索条数
- 检索到但未使用的比例

## 7. 这个 RAG 设计和普通问答 RAG 的不同

普通问答 RAG 更关注“能不能答上来”，医疗 RAG 更关注：

- 能不能给出有边界的答案
- 能不能支持分诊和检查建议
- 能不能在用药和危险信号上提供可靠证据

所以医疗 RAG 的重点不是堆更多文档，而是把：

- 指南
- 药典
- 科普
- 流程

按任务拆开来用。
