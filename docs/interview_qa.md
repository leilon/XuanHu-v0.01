# Interview Q&A (Medical Multi-Agent + 7B)

## Q1: Why choose 7B instead of larger models?
A:
- Cost/performance tradeoff for internship-scale infra
- QLoRA lets us iterate quickly and do continual learning
- For tool-augmented tasks (RAG + structured tools), 7B can meet product targets with good engineering

## Q2: How do you implement long-term memory with QLoRA?
A:
- Split memory into two layers:
  - External memory: profile + episodic memory in store
  - Parametric memory: task adapters trained via QLoRA
- At inference, memory fusion injects retrieved profile/episodes into prompt and selects task adapter from Adapter Bank

## Q3: Why multi-agent architecture?
A:
- Medical workflow naturally decomposes into intake, triage, medication, report interpretation
- Specialized agents improve controllability and debugging
- Orchestrator enables dynamic planning and tool-call routing

## Q4: How do you avoid hallucinations in medical responses?
A:
- RAG with citation-required generation
- Safety guardrails for emergency signals
- Benchmark with grounding and safety recall metrics
- Preference optimization on agentic trajectories

## Q5: What is your Agentic RL strategy?
A:
- Build preference pairs from benchmark + real dialogue logs
- Optimize policy adapter with DPO as practical RLHF stage
- Reward proxies:
  - task completion
  - safety token recall
  - grounding citation quality
  - tool-call efficiency

## Q6: How is multimodal handled?
A:
- Separate report/image specialist branch based on VL-7B
- Extract report findings, summarize abnormalities, and route to triage/medication modules
- Keep output structured with explicit risk note and next-step recommendation

## Q7: How do you evaluate before deployment?
A:
- Horizontal benchmark vs baseline single-agent
- Case-level scoring:
  - medical correctness
  - safety
  - grounding
- Regression checks across model and adapter versions

## Q8: If GPU budget is limited, what do you do first?
A:
- Stage 1: 1xA100 for SFT + benchmark iteration
- Stage 2: add RL preference tuning after data quality stabilizes
- Stage 3: scale to 2-4 GPUs only when throughput becomes bottleneck

