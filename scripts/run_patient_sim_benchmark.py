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
from medagent.benchmark.patient_simulator import PatientSimulator, SimConfig
from medagent.langgraph_orchestrator import LangChainOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn benchmark with API/rule-based patient simulator")
    parser.add_argument("--scenarios", default="data/sim_patient_cases.json")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--report-file", default="")
    parser.add_argument("--sim-model", default="gpt-4o-mini")
    parser.add_argument("--sim-base-url", default="")
    parser.add_argument("--sim-api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    with open(args.scenarios, "r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    simulator = PatientSimulator(
        SimConfig(
            model=args.sim_model,
            base_url=args.sim_base_url,
            api_key_env=args.sim_api_key_env,
        )
    )
    orchestrator = LangChainOrchestrator()
    evaluator = BenchmarkEvaluator()
    scores = []
    report_rows = []

    for scenario in scenarios:
        history: list[dict] = []
        visit_id = None
        user_text = scenario["opening"]
        final_result: dict | None = None

        for turn_idx in range(1, args.max_turns + 1):
            final_result = orchestrator.run_visit_turn(
                user_id=f"sim_{scenario['id']}",
                user_text=user_text,
                visit_id=visit_id,
                age=scenario.get("age"),
                sex=scenario.get("sex"),
                image_path=scenario.get("image_path"),
            )
            visit_id = final_result["visit_id"]
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": final_result["response"]})

            if final_result.get("visit_completed"):
                break

            user_text = simulator.respond(
                scenario,
                history,
                latest_agent_reply=final_result["response"],
                visit_state=final_result.get("visit_record"),
            )

        final_result = final_result or {
            "response": "",
            "visit_completed": False,
            "visit_record": {},
            "triage_label": "",
        }
        score = evaluator.score(
            case_id=scenario["id"],
            strategy="multiturn_langgraph_patient_sim",
            prediction=final_result["response"],
            expected=scenario["expected"],
        )
        scores.append(score)
        report_rows.append(
            {
                "id": scenario["id"],
                "history": history,
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
            f"{scenario['id']}\toverall={score.overall:.3f}\tmedical={score.score_medical:.3f}"
            f"\tsafety={score.score_safety:.3f}\tgrounding={score.score_grounding:.3f}"
        )

    print(f"[summary] avg_overall={mean(item.overall for item in scores):.3f}")

    if args.report_file:
        output_path = Path(args.report_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] wrote {output_path}")


if __name__ == "__main__":
    main()
