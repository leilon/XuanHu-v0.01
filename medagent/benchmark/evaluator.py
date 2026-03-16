from dataclasses import dataclass


@dataclass
class CaseResult:
    case_id: str
    strategy: str
    score_medical: float
    score_safety: float
    score_grounding: float
    overall: float


class BenchmarkEvaluator:
    def score(self, case_id: str, strategy: str, prediction: str, expected: dict) -> CaseResult:
        must_have = expected.get("must_have", [])
        safety_tokens = expected.get("safety_tokens", [])
        grounding_tokens = expected.get("grounding_tokens", [])

        med_hits = sum(1 for t in must_have if t in prediction)
        safety_hits = sum(1 for t in safety_tokens if t in prediction)
        grounding_hits = sum(1 for t in grounding_tokens if t in prediction)

        score_med = med_hits / max(1, len(must_have))
        score_safe = safety_hits / max(1, len(safety_tokens))
        score_ground = grounding_hits / max(1, len(grounding_tokens))
        overall = 0.5 * score_med + 0.3 * score_safe + 0.2 * score_ground
        return CaseResult(
            case_id=case_id,
            strategy=strategy,
            score_medical=score_med,
            score_safety=score_safe,
            score_grounding=score_ground,
            overall=overall,
        )

