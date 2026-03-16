# Agentic RL Synthesis

## Why the old script was not good enough

- It used the current local orchestrator as the "chosen" policy, so it could only reproduce the weaknesses already in the prototype.
- It used a trivial generic string as the "rejected" policy, which makes the preference task unrealistically easy.
- It did not synthesize multi-turn trajectories, tool calls, or safety failures.
- It did not judge whether the pair had a meaningful quality gap.

## Better synthesis recipe

Use a stronger OpenAI-compatible teacher model to generate pairwise data in four stages:

1. Build a hidden case packet from a seed prompt.
2. Generate an expert trajectory with follow-up questions, tool calls, and a grounded final answer.
3. Generate a flawed but plausible trajectory for a specific failure mode.
4. Judge the pair and only keep it if the quality margin is large enough.

## Failure modes to synthesize

- `premature_answer`
- `missed_red_flag`
- `no_tool_use`
- `no_grounding`
- `missed_history`

These are much closer to the real mistakes we care about in a medical agent than generic weak responses.

## Recommended data mix for 7B

- 60%: high-quality SFT distilled trajectories
- 25%: synthetic agentic preference pairs from the new pipeline
- 15%: public reward / preference sets after manual filtering

Public reward data that is still useful:

- `shibing624/medical/reward/train.json`
  - Chinese medical pairwise reward seed
- `liyucheng/zhihu_rlhf_3k`
  - Only use a filtered health subset, not the full set

## Output format

Each synthesized sample becomes:

- `prompt`
- `chosen`
- `rejected`
- `meta.case`
- `meta.failure_mode`
- `meta.judge`

This format is directly usable for DPO-style fine-tuning.
