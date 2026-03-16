#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from statistics import mean
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.benchmark.evaluator import BenchmarkEvaluator
from medagent.benchmark.patient_simulator import PatientSimulator
from medagent.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark with LLM/API patient simulator")
    parser.add_argument("--scenarios", default="data/sim_patient_cases.json")
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    with open(args.scenarios, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    sim = PatientSimulator()
    orch = Orchestrator()
    evaluator = BenchmarkEvaluator()
    scores = []

    for sc in scenarios:
        history = []
        question = sc["opening"]
        final_response = ""
        for turn in range(args.max_turns):
            final_response = orch.run(user_id=f"sim_{sc['id']}", user_text=question)
            history.append({"assistant": final_response})
            question = sim.respond(sc, history)
            history.append({"user": question})

        score = evaluator.score(
            case_id=sc["id"],
            strategy="multi_agent_with_patient_sim",
            prediction=final_response,
            expected=sc["expected"],
        )
        scores.append(score)
        print(f"{sc['id']}\toverall={score.overall:.3f}\tmedical={score.score_medical:.3f}\tsafety={score.score_safety:.3f}\tgrounding={score.score_grounding:.3f}")

    print(f"[summary] avg_overall={mean(s.overall for s in scores):.3f}")


if __name__ == "__main__":
    main()
