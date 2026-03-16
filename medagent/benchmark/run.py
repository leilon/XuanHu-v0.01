import argparse
import json
from statistics import mean

from medagent.benchmark.evaluator import BenchmarkEvaluator, CaseResult
from medagent.orchestrator import Orchestrator


def run_baseline(question: str) -> str:
    # Simulated single-agent baseline for horizontal comparison.
    return f"建议补液休息并观察，如持续不适请就医。问题：{question}"


def run_multi_agent(question: str) -> str:
    orchestrator = Orchestrator()
    return orchestrator.run(user_id="benchmark_user", user_text=question)


def print_results(results: list[CaseResult]) -> None:
    print("case_id\tstrategy\tmedical\tsafety\tgrounding\toverall")
    for item in results:
        print(
            f"{item.case_id}\t{item.strategy}\t"
            f"{item.score_medical:.2f}\t{item.score_safety:.2f}\t"
            f"{item.score_grounding:.2f}\t{item.overall:.2f}"
        )

    for strategy in sorted(set(r.strategy for r in results)):
        subset = [r for r in results if r.strategy == strategy]
        print(
            f"[summary] {strategy}: overall={mean(r.overall for r in subset):.3f}, "
            f"medical={mean(r.score_medical for r in subset):.3f}, "
            f"safety={mean(r.score_safety for r in subset):.3f}, "
            f"grounding={mean(r.score_grounding for r in subset):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAgent benchmark comparison")
    parser.add_argument("--dataset", required=True, help="Path to benchmark cases")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        cases = json.load(f)

    evaluator = BenchmarkEvaluator()
    results: list[CaseResult] = []
    for item in cases:
        case_id = item["id"]
        question = item["question"]
        expected = item["expected"]

        baseline_pred = run_baseline(question)
        ma_pred = run_multi_agent(question)

        results.append(evaluator.score(case_id, "baseline_single_agent", baseline_pred, expected))
        results.append(evaluator.score(case_id, "multi_agent_orchestrator", ma_pred, expected))

    print_results(results)


if __name__ == "__main__":
    main()

