#!/usr/bin/env python
"""
Synthesize higher-quality agentic RL preference pairs with a strong teacher model.

Pipeline:
1. Build a structured hidden case packet from a seed prompt.
2. Generate an expert multi-turn trajectory with tool use.
3. Generate a flawed but plausible trajectory for a specific failure mode.
4. Judge the pair and keep only examples with a clear preference margin.

This script expects an OpenAI-compatible API for real data generation.
Use --dry-run to inspect the pipeline without making API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FAILURE_MODES = [
    "premature_answer",
    "missed_red_flag",
    "no_tool_use",
    "no_grounding",
    "missed_history",
]


def _read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _load_json_response(text: str) -> dict[str, Any]:
    text = _strip_fence(text)
    return json.loads(text)


class OpenAICompatClient:
    # Minimal OpenAI-compatible wrapper so we can swap providers
    # (OpenAI, Doubao-compatible gateway, local proxy) without
    # changing the synthesis pipeline logic below.
    def __init__(self, model: str, api_key_env: str, base_url: str = "") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url

    @property
    def available(self) -> bool:
        return bool(os.getenv(self.api_key_env))

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
        if not self.available:
            raise RuntimeError(
                f"Missing API key in env var {self.api_key_env}. "
                "Set it or use --dry-run."
            )

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("Install openai to run synthetic generation.") from exc

        kwargs: dict[str, Any] = {"api_key": os.getenv(self.api_key_env)}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)

        last_error: Exception | None = None
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content
                if not content:
                    raise RuntimeError("Empty completion content.")
                return content
            except Exception as exc:
                last_error = exc
                time.sleep(1.5)
        raise RuntimeError(f"Completion failed after retries: {last_error}")


def _build_fallback_case(prompt: str) -> dict[str, Any]:
    # Dry-run mode still needs a realistic hidden case packet so we can
    # test the downstream data flow without external API calls.
    red_flags: list[str] = []
    likely_tools = ["rag_guideline", "drug_knowledge"]
    if any(k in prompt for k in ("胸痛", "呼吸困难", "抽搐", "便血")):
        red_flags.append("possible_emergency")
    if any(k in prompt for k in ("报告", "化验", "白细胞", "CT", "MRI", "影像")):
        likely_tools.append("report_parser")
    must_ask = ["duration", "allergy_history", "current_meds"]
    return {
        "opening": prompt,
        "profile": {"age": "", "sex": ""},
        "hidden_facts": [
            "duration is unclear at first",
            "allergy history is not stated unless asked",
        ],
        "red_flags": red_flags,
        "must_ask": must_ask,
        "likely_tools": likely_tools,
        "evidence_snippets": [
            "guideline: ask duration and risk factors before recommending treatment",
            "drug_knowledge: mention contraindications and interactions",
        ],
        "good_disposition": "ask key follow-up questions, use tools when needed, and escalate if red flags exist",
        "difficulty": "medium",
    }


def build_case_packet(
    client: OpenAICompatClient,
    prompt: str,
    reference_answer: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    # Step 1: convert a raw user query into a hidden structured case.
    # This is the key difference from naive "rewrite the answer" data
    # generation: later stages optimize over the same latent case state.
    if dry_run or not client.available:
        return _build_fallback_case(prompt)

    system_prompt = (
        "You create hidden medical simulation cases for agentic RL data generation. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Seed user question:
{prompt}

Optional reference answer:
{reference_answer}

Return JSON with fields:
- opening: string
- profile: object with age, sex
- hidden_facts: list of strings
- red_flags: list of strings
- must_ask: list of strings
- likely_tools: list chosen from [rag_guideline, drug_knowledge, report_parser, triage_router]
- evidence_snippets: list of short strings
- good_disposition: short string
- difficulty: easy|medium|hard

Requirements:
- Keep it realistic for a consumer-facing medical assistant.
- Use Chinese if the seed prompt is Chinese.
- Do not reveal the diagnosis as a certainty.
""".strip()
    return _load_json_response(client.complete(system_prompt, user_prompt, temperature=0.2))


def _fallback_expert_trajectory(case: dict[str, Any]) -> dict[str, Any]:
    opening = case["opening"]
    turns = [
        {
            "speaker": "assistant",
            "kind": "ask",
            "content": "请先补充症状持续时间、体温范围、药物过敏史和当前用药情况。",
        },
        {
            "speaker": "user",
            "kind": "reply",
            "content": "目前信息还不完整，需要进一步追问。",
        },
        {
            "speaker": "assistant",
            "kind": "tool_call",
            "tool_name": "rag_guideline",
            "content": f"query={opening}",
        },
        {
            "speaker": "tool",
            "kind": "tool_result",
            "tool_name": "rag_guideline",
            "content": "; ".join(case.get("evidence_snippets", [])),
        },
    ]
    final_answer = (
        "基于当前信息，建议先补全关键病史后再做更具体建议；如出现高危信号请立即线下就医。"
        "同时需要结合指南证据和禁忌信息判断是否适合用药。"
    )
    return {"turns": turns, "final_answer": final_answer}


def generate_expert_trajectory(
    client: OpenAICompatClient,
    case: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    # Step 2: synthesize the preferred trajectory. We want a multi-turn,
    # tool-aware consultation trace, not just a polished final response.
    if dry_run or not client.available:
        return _fallback_expert_trajectory(case)

    system_prompt = (
        "You generate expert trajectories for a medical assistant agent. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Hidden case packet:
{json.dumps(case, ensure_ascii=False, indent=2)}

Return JSON with:
- turns: list of objects with fields:
  - speaker: assistant|user|tool
  - kind: ask|reply|tool_call|tool_result
  - content: string
  - tool_name: optional string
- final_answer: string

Requirements:
- Create a realistic multi-turn dialogue of 4 to 8 turns.
- The assistant should ask for missing critical history before giving a final answer.
- Use at least one tool if likely_tools is not empty.
- If red_flags is non-empty, the final answer must escalate appropriately.
- The final answer must be evidence-grounded and action-oriented.
""".strip()
    return _load_json_response(client.complete(system_prompt, user_prompt, temperature=0.4))


def _fallback_negative_trajectory(case: dict[str, Any], expert: dict[str, Any], failure_mode: str) -> dict[str, Any]:
    turns = [
        {
            "speaker": "assistant",
            "kind": "ask",
            "content": "先观察一下，注意休息。",
        }
    ]
    final_answer = "先多喝水休息，暂时不用进一步处理。"
    if failure_mode == "no_grounding":
        final_answer = "考虑问题不大，按经验先观察就可以。"
    elif failure_mode == "premature_answer":
        final_answer = "初步判断就是普通小问题，不需要再追问。"
    elif failure_mode == "missed_red_flag":
        final_answer = "可以继续在家观察，不必着急就医。"
    return {"turns": turns, "final_answer": final_answer}


def generate_negative_trajectory(
    client: OpenAICompatClient,
    case: dict[str, Any],
    expert: dict[str, Any],
    failure_mode: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    # Step 3: synthesize a plausible but worse trajectory. The failure must
    # be specific (missed history, no grounding, etc.), otherwise DPO pairs
    # become too easy and teach little about agent behavior.
    if dry_run or not client.available:
        return _fallback_negative_trajectory(case, expert, failure_mode)

    system_prompt = (
        "You generate flawed but plausible medical agent trajectories for preference training. "
        "Return valid JSON only."
    )
    user_prompt = f"""
Hidden case packet:
{json.dumps(case, ensure_ascii=False, indent=2)}

Reference expert trajectory:
{json.dumps(expert, ensure_ascii=False, indent=2)}

Failure mode:
{failure_mode}

Return JSON with:
- turns: list of objects with fields speaker, kind, content, tool_name(optional)
- final_answer: string

Requirements:
- The trajectory should still look plausible, not cartoonishly bad.
- It must be worse than the expert trajectory specifically because of the requested failure mode.
- Example failure modes:
  - premature_answer: answers before collecting key history
  - missed_red_flag: does not escalate despite danger signs
  - no_tool_use: skips an obviously needed tool
  - no_grounding: gives advice without evidence or citations
  - missed_history: forgets allergy, pregnancy, medication, or chronic disease history
""".strip()
    return _load_json_response(client.complete(system_prompt, user_prompt, temperature=0.6))


def _render_trajectory(case: dict[str, Any], traj: dict[str, Any]) -> str:
    # Flatten structured trajectories into a DPO-friendly text format while
    # preserving turn roles, tool calls, and the final answer.
    lines = [
        f"[opening] {case.get('opening', '').strip()}",
        f"[difficulty] {case.get('difficulty', 'unknown')}",
    ]
    for turn in traj.get("turns", []):
        speaker = turn.get("speaker", "assistant")
        kind = turn.get("kind", "reply")
        tool_name = turn.get("tool_name", "")
        label = f"{speaker}/{kind}"
        if tool_name:
            label += f"/{tool_name}"
        lines.append(f"[{label}] {turn.get('content', '').strip()}")
    lines.append(f"[final_answer] {traj.get('final_answer', '').strip()}")
    return "\n".join(lines)


def _score_text(case: dict[str, Any], text: str) -> dict[str, float]:
    # Dry-run heuristic judge. The real run should use a stronger judge model,
    # but this gives us a deterministic fallback for local testing.
    score = {
        "safety": 0.0,
        "coverage": 0.0,
        "grounding": 0.0,
        "tool_use": 0.0,
    }
    lower = text.lower()

    if any(flag in lower for flag in ("就医", "急诊", "emergency", "立即")) or not case.get("red_flags"):
        score["safety"] = 1.0
    must_ask = case.get("must_ask", [])
    if must_ask:
        hit = sum(1 for item in must_ask if item.lower() in lower or item in text)
        score["coverage"] = hit / max(1, len(must_ask))
    else:
        score["coverage"] = 1.0

    if any(token in lower for token in ("guideline", "evidence", "引用", "依据", "tool_result")):
        score["grounding"] = 1.0
    if any(tool.lower() in lower for tool in case.get("likely_tools", [])):
        score["tool_use"] = 1.0

    score["overall"] = (
        0.35 * score["safety"]
        + 0.30 * score["coverage"]
        + 0.20 * score["grounding"]
        + 0.15 * score["tool_use"]
    )
    return score


def judge_pair(
    client: OpenAICompatClient,
    case: dict[str, Any],
    chosen_text: str,
    rejected_text: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    # Step 4: only keep pairs with a meaningful preference signal.
    # Without a judge stage, synthesis pipelines keep too many noisy or
    # near-tie examples that make preference training unstable.
    if dry_run or not client.available:
        chosen_score = _score_text(case, chosen_text)
        rejected_score = _score_text(case, rejected_text)
        verdict = "chosen" if chosen_score["overall"] >= rejected_score["overall"] else "rejected"
        return {
            "verdict": verdict,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }

    system_prompt = "You are a strict judge for medical agent preference data. Return valid JSON only."
    user_prompt = f"""
Hidden case packet:
{json.dumps(case, ensure_ascii=False, indent=2)}

Candidate A:
{chosen_text}

Candidate B:
{rejected_text}

Return JSON with:
- verdict: A|B
- rationale: short string
- scores:
  - A: object with safety, coverage, grounding, tool_use, overall in [0,1]
  - B: object with safety, coverage, grounding, tool_use, overall in [0,1]

Judge criteria:
- Safety correctness is the highest priority.
- Prefer trajectories that ask for missing critical history.
- Prefer grounded tool-backed answers.
- Penalize premature diagnosis and skipped escalation.
""".strip()
    result = _load_json_response(client.complete(system_prompt, user_prompt, temperature=0.1))
    return {
        "verdict": "chosen" if result.get("verdict") == "A" else "rejected",
        "chosen_score": result.get("scores", {}).get("A", {}),
        "rejected_score": result.get("scores", {}).get("B", {}),
        "rationale": result.get("rationale", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize agentic RL preference pairs")
    parser.add_argument("--seeds-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--seed-field", default="input")
    parser.add_argument("--answer-field", default="output")
    parser.add_argument("--teacher-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)
    teacher = OpenAICompatClient(args.teacher_model, args.api_key_env, args.base_url)
    judge = OpenAICompatClient(args.judge_model or args.teacher_model, args.api_key_env, args.base_url)

    seeds = list(_read_jsonl(args.seeds_file))
    random.shuffle(seeds)
    seeds = seeds[: args.max_samples]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(seeds):
            prompt = str(item.get(args.seed_field, "")).strip()
            reference_answer = str(item.get(args.answer_field, "")).strip()
            if not prompt:
                continue

            case = build_case_packet(teacher, prompt, reference_answer, dry_run=args.dry_run)
            expert = generate_expert_trajectory(teacher, case, dry_run=args.dry_run)
            failure_mode = random.choice(DEFAULT_FAILURE_MODES)
            negative = generate_negative_trajectory(
                teacher, case, expert, failure_mode, dry_run=args.dry_run
            )

            chosen_text = _render_trajectory(case, expert)
            rejected_text = _render_trajectory(case, negative)
            judged = judge_pair(judge, case, chosen_text, rejected_text, dry_run=args.dry_run)

            chosen_score = float(judged["chosen_score"].get("overall", 0.0))
            rejected_score = float(judged["rejected_score"].get("overall", 0.0))
            if judged["verdict"] != "chosen":
                chosen_text, rejected_text = rejected_text, chosen_text
                chosen_score, rejected_score = rejected_score, chosen_score

            # Margin filtering drops ambiguous pairs so the retained data is
            # sharper for DPO/ORPO-style optimization on smaller 7B models.
            if chosen_score - rejected_score < args.min_margin:
                continue

            row = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "meta": {
                    "source": "synthetic_agentic_v2",
                    "case": case,
                    "failure_mode": failure_mode,
                    "judge": judged,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            if kept % 50 == 0:
                print(f"[progress] kept {kept} examples", flush=True)

    print(f"[ok] wrote {kept} synthetic preference pairs to {out_path}")


if __name__ == "__main__":
    main()
