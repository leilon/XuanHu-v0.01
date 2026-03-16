# 当前待办清单

## 1. Git 与仓库管理

- 统一本地、AutoDL 工作目录和后续 GitHub 仓库状态
- 在 GitHub 上创建正式仓库，优先考虑：
  - `QingNang-ClinicOS`
  - `QingNang-MedAgent`
- 给项目和各 Agent 做统一命名方案，避免与现有医疗模型重名

## 2. 模型与运行环境

- 下载并整理当前真正要用的基模
  - 当前文本主干：`Qwen2.5-7B-Instruct`
  - 当前视觉主干：`HuatuoGPT-Vision-7B-Qwen2.5VL`
- 清理不再使用的旧视觉基模和无效缓存
- 确认 AutoDL 实例切换到有 GPU 的容器后再启动训练
- 增加模型状态检查脚本和下载恢复逻辑

## 3. 临床首程问诊逻辑

- 重构 `clinical_pathway.py`
  - 从关键词规则升级为“症状分类 + 问诊树 + 红旗规则 + 分诊规则”四层结构
- 当前 `IntentRouter` 仍然是规则型占位实现
  - 后续要升级为“规则 + 小模型 / API judge”联合路由
  - 需要加入更多真实用户表达、模糊主诉、知识问答与首程问诊的边界判别
- 所有首程病例与训练样本都要从“信息不完整的真实主诉”开始
  - 不能默认用户一上来就提供体温、血压、症状持续时间
  - 要训练 Agent 主动追问缺失的关键生命体征、时间线和危险信号
  - 合成病例时要覆盖“说不清楚、不主动报数值、只说难受”的用户表达
- 把首程问诊逻辑拆成独立配置文件
  - `symptom_taxonomy`
  - `question_graph`
  - `red_flag_rules`
  - `disposition_rules`
- 补充更多真实首程问诊场景：
  - 呼吸道
  - 胸痛
  - 腹痛
  - 泌尿/妇科
  - 神经系统
  - 皮疹/过敏

## 4. 长期记忆与遗忘策略

- 将长期记忆拆成两层：
  - 医学知识 RAG
  - 患者个人记忆库
- 明确什么需要长期存：
  - 性别
  - 年龄
  - 慢病
  - 手术史
  - 药物过敏史
  - 严重不良反应史
  - 长期用药
  - 关键检查异常
  - 既往相似发作的摘要
- 明确什么只短期保存：
  - 本次急性症状细节
  - 流行病接触史
  - 旅行史
  - 妊娠状态 / 末次月经
  - 单次 episode 的临时参数
- 增加 TTL / 置信度 / 冲突解决 / 用户纠错逻辑
- 把“首程问诊总结”纳入长期记忆设计
  - 不直接存原始长对话
  - 优先存结构化首程摘要
  - 该摘要建议由强 API 生成，而不是完全依赖本地 7B

## 5. 强 API 接口与复杂问题升级

- 设计统一的 `LLMRouter`
- 约定哪些复杂情况升级到强 API：
  - 高风险且歧义大的多系统症状
  - 长链路工具调用
  - 报告 + 用药 + 既往史同时冲突
  - 本地 7B 多轮后仍低置信
  - 困难视觉病例
- 把强 API 用于：
  - 首程问诊摘要生成
  - memory salience judge
  - synthetic RL judge
  - patient simulator

## 6. RAG 主线

- 彻底重写 `medagent/services/rag.py`
- 将目前的硬编码文档替换为真实检索链路：
  - 文档下载
  - 清洗
  - chunk
  - embedding
  - vector index
  - rerank
  - 引用生成
- 分开两套索引：
  - 医学知识索引
  - 患者个体记忆索引
- 中文主语料至少做到 `1GB+` 原始规模，更理想是 `2GB - 8GB`
- 下载更可靠的原始资料：
  - 药品标签
  - 官方健康教育资料
  - 中文临床参考手册
  - 中文教材 / 中文医学百科 / 中文医患问答
  - 官方指南或可靠 guideline 文档
- 评估当前 `guidelines_medqa_qa_v1` 是否只适合作弱知识补充

## 7. RAG 文档处理细节

- 研究可用公开源：
  - DailyMed
  - openFDA drug label API
  - MedlinePlus
  - `hf-mirror` 上的中文医疗语料
  - `MSD 手册中文专业版`
- 当前重点中文数据池：
  - `FreedomIntelligence/huatuo_encyclopedia_qa`
  - `FreedomIntelligence/huatuo_knowledge_graph_qa`
  - `FreedomIntelligence/Huatuo26M-Lite`
  - `shibing624/medical`
  - `wangrongsheng/cMedQA-V2.0`
  - `ticoAg/Chinese-medical-dialogue`
- 制定 chunk metadata：
  - `source_id`
  - `source_type`
  - `disease_tags`
  - `drug_tags`
  - `updated_at`
  - `evidence_level`
- 确定 embedding / reranker 方案
  - 初版考虑 `bge-m3` + `bge-reranker`

## 8. QLoRA 主线

- 明确 QLoRA 的核心用途，不把它只当“炫技点”
- 重新设计文本 SFT 数据桶，不再把首程问诊和科普解释混在一起
  - `BianQue-Intake`：短轮次、一问一答、首轮追问风格
  - `LiShiZhen-Education`：长解释、科普、指标释义、患者友好说明
- 等中文 RAG 主语料 chunk 完成后，重新定义文本 SFT 的输入输出风格
  - `BianQue-Intake` 偏首程追问
  - `LiShiZhen-Education` 偏引用知识后的解释与安抚
- 目前更适合的用途：
  - 任务 adapter
  - 周期性持续学习
  - 强 API teacher 蒸馏
  - 场景 adapter
  - 轻量 critic / extractor
- 不建议把 QLoRA 作为主要 test-time training 手段
- 后续需要补：
  - eval split
  - checkpoint resume
  - accelerate / deepspeed 配置
  - 更细的 adapter routing

## 9. Agentic-RL 主线

- 继续完善合成数据方案
- 用“临床首程问诊逻辑”约束 `chosen / rejected` 轨迹生成
- 把“强 API 生成高质量首程问诊摘要/病历摘要”接入数据合成链路
- 增加失败模式：
  - 漏问慢病
  - 漏问长期用药
  - 男性误问怀孕
  - 育龄女性漏问妊娠
  - 分诊级别错误
- 逐步弱化 `build_agentic_rl_data.py` 的占位作用

## 10. 多模态主线

- 完成 VL 模型下载
- 优先做报告/化验单理解，不急着做复杂影像诊断
- 补医学报告与 VQA 类数据集整理
- 明确多模态输入输出格式

## 11. Benchmark 与评测

- 重写 benchmark evaluator
  - 不能只靠 token overlap
- 增加多轮问诊评测：
  - 关键病史覆盖率
  - 危险信号召回率
  - 记忆命中率
  - 不合理追问率
- 继续接入 LLM API 作为病人模拟器
- 增加长期记忆场景和特殊人群场景

## 12. 当前明确仍是占位符的模块

- `medagent/services/rag.py`
- `medagent/services/tools.py`
- `medagent/agents/report.py`
- `scripts/build_agentic_rl_data.py`
- `medagent/benchmark/evaluator.py`
- `medagent/benchmark/run.py`
- `medagent/benchmark/patient_simulator.py`
- `medagent/services/clinical_pathway.py` 目前仍偏规则模板

## 13. 文档与展示

- 继续保持中文主文档优先
- 准备 GitHub 首页展示版 README
- 增加架构图、首程问诊流程图和训练路线图
- 为面试准备一页“项目亮点”总结
