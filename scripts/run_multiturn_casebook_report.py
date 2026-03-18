#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.benchmark.evaluator import BenchmarkEvaluator
from medagent.benchmark.patient_simulator import PatientSimulator
from medagent.config import AppConfig
from medagent.langgraph_orchestrator import LangChainOrchestrator


def _write_markdown(report_rows: list[dict], output_path: Path) -> None:
    branding = AppConfig().branding
    all_turn_latencies = [
        turn["elapsed_sec"]
        for row in report_rows
        for turn in row.get("turn_trace", [])
        if turn.get("elapsed_sec") is not None
    ]
    lines = [
        f"# {branding.project_name_zh}多轮真实 Case 报告",
        "",
        "这份报告基于 10 个预设病人场景，在单进程中调用本地 `HuatuoGPT-Vision-7B-Qwen2.5VL` 生成。",
        "",
    ]

    if report_rows:
        lines.extend(
            [
                "## 总览",
                f"- `case_count`: {len(report_rows)}",
                f"- `avg_overall`: {mean(item['score']['overall'] for item in report_rows):.3f}",
                f"- `avg_medical`: {mean(item['score']['medical'] for item in report_rows):.3f}",
                f"- `avg_safety`: {mean(item['score']['safety'] for item in report_rows):.3f}",
                f"- `avg_grounding`: {mean(item['score']['grounding'] for item in report_rows):.3f}",
                f"- `avg_turns`: {mean(len(item['history']) // 2 for item in report_rows):.2f}",
                f"- `avg_turn_latency_sec`: {mean(all_turn_latencies):.3f}" if all_turn_latencies else "- `avg_turn_latency_sec`: 0.000",
                "",
            ]
        )

    for row in report_rows:
        case = row["case"]
        result = row["final_result"]
        turn_trace = row.get("turn_trace", [])
        lines.extend(
            [
                f"## {case['id']}｜{case['title']}",
                f"- `severity`: {case['severity']}",
                f"- `education_level`: {case['education_level']}",
                f"- `speaking_style`: {case['speaking_style']}",
                f"- `overall`: {row['score']['overall']:.3f}",
                f"- `medical`: {row['score']['medical']:.3f}",
                f"- `safety`: {row['score']['safety']:.3f}",
                f"- `grounding`: {row['score']['grounding']:.3f}",
                f"- `visit_completed`: {result['visit_completed']}",
                f"- `stop_reason`: {result['stop_reason'] or 'continue'}",
                f"- `avg_turn_latency_sec`: {mean(turn['elapsed_sec'] for turn in turn_trace):.3f}" if turn_trace else "- `avg_turn_latency_sec`: 0.000",
                "",
                "### 病人背景",
                f"- 年龄/性别：{case['age']} / {case['sex']}",
                f"- 既往背景：{case['background']['history']}",
                f"- 过敏情况：{case['background']['allergies']}",
                f"- 就诊目标：{case['background']['goal']}",
                "",
                "### 对话记录",
            ]
        )
        for turn in turn_trace:
            lines.append(f"- 患者：{turn['user']}")
            lines.append(f"- {branding.project_name_zh}：{turn['assistant']}")
            lines.append(f"- 本轮耗时：{turn['elapsed_sec']:.3f}s")
        lines.extend(
            [
                "",
                "### 最终结果",
                f"- 最终回复：{result['response']}",
                f"- 初步判断：{result.get('preliminary_assessment', '')}",
                f"- 分诊标签：{result.get('triage_label', '')}",
                f"- 建议检查：{'、'.join(result.get('recommended_tests', [])) or '暂无'}",
                "",
                "### 自动首程摘要",
                "```text",
                result.get("visit_record", {}).get("human_readable_summary", ""),
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10-case multiturn report with local model and simulated patients.")
    parser.add_argument("--cases", default="data/multiturn_casebook_10.json")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--json-out", default="docs/reports/multiturn_casebook_report.json")
    parser.add_argument("--md-out", default="docs/reports/multiturn_casebook_report_zh.md")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    simulator = PatientSimulator()
    print(f"[init] loaded {len(cases)} cases", flush=True)
    orchestrator = LangChainOrchestrator()
    print("[init] orchestrator ready", flush=True)
    evaluator = BenchmarkEvaluator()
    report_rows: list[dict] = []

    for case in cases:
        simulator.reset_case(case["id"])
        print(f"[case] start {case['id']} {case['title']}", flush=True)
        history: list[dict] = []
        turn_trace: list[dict] = []
        visit_id = None
        user_text = case["opening"]
        final_result: dict | None = None

        for turn_idx in range(1, args.max_turns + 1):
            print(f"[case] {case['id']} turn={turn_idx}", flush=True)
            final_result = orchestrator.run_visit_turn(
                user_id=f"casebook_{case['id']}",
                user_text=user_text,
                visit_id=visit_id,
                age=case.get("age"),
                sex=case.get("sex"),
            )
            visit_id = final_result["visit_id"]
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": final_result["response"]})
            turn_trace.append(
                {
                    "turn_index": turn_idx,
                    "user": user_text,
                    "assistant": final_result["response"],
                    "elapsed_sec": float(final_result.get("elapsed_sec", 0.0)),
                }
            )
            if final_result.get("visit_completed"):
                print(f"[case] {case['id']} completed stop={final_result.get('stop_reason', '')}", flush=True)
                break
            user_text = simulator.respond(
                case,
                history,
                latest_agent_reply=final_result["response"],
                visit_state=final_result.get("visit_record"),
            )

        final_result = final_result or {
            "response": "",
            "visit_completed": False,
            "stop_reason": "",
            "triage_label": "",
            "preliminary_assessment": "",
            "recommended_tests": [],
            "visit_record": {},
        }
        score = evaluator.score(
            case_id=case["id"],
            strategy="multiturn_casebook",
            prediction=final_result["response"],
            expected=case["expected"],
        )
        report_rows.append(
            {
                "case": case,
                "history": history,
                "turn_trace": turn_trace,
                "final_result": final_result,
                "score": {
                    "overall": score.overall,
                    "medical": score.score_medical,
                    "safety": score.score_safety,
                    "grounding": score.score_grounding,
                },
            }
        )
        print(
            f"{case['id']}\toverall={score.overall:.3f}\tmedical={score.score_medical:.3f}"
            f"\tsafety={score.score_safety:.3f}\tgrounding={score.score_grounding:.3f}"
        )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(report_rows, md_out)

    print(f"[json] wrote {json_out}")
    print(f"[md] wrote {md_out}")


if __name__ == "__main__":
    main()
