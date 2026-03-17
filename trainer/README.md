# trainer 目录说明

`trainer/` 只放训练相关脚本，不再和部署、RAG、评测脚本混在一起。

当前目录约定：

- `trainer/data/`
  - 训练数据下载与统一清洗
  - 输入是 AutoDL 数据盘上的原始数据集
  - 输出是任务化 SFT / RL 可以直接消费的中间产物

- `trainer/text/`
  - 文本 SFT 数据准备与训练
  - 当前重点是 `BianQue-Intake`

- `trainer/vision/`
  - 视觉 SFT 数据校验与训练
  - 当前重点是 `CangGong-Report`

- `trainer/rl/`
  - Agentic-RL 偏好数据构建、过滤、训练

- `trainer/core/`
  - 通用 QLoRA 训练入口
  - 更适合做实验、回放和持续学习

## 数据路径约定

原始下载数据：
- `/root/autodl-tmp/medagent/datasets/sft`
- `/root/autodl-tmp/medagent/datasets/rl`

清洗后的任务数据：
- `/root/autodl-tmp/medagent/datasets/curated/bianque_intake`
- `/root/autodl-tmp/medagent/datasets/curated/lishizhen_education`
- `/root/autodl-tmp/medagent/datasets/curated/canggong_vision`

训练产物：
- `/root/autodl-tmp/medagent/outputs/adapters`
