import argparse
import json

from medagent.config import AppConfig
from medagent.langgraph_orchestrator import LangChainOrchestrator


def main() -> None:
    branding = AppConfig().branding
    parser = argparse.ArgumentParser(
        description=f"{branding.project_name_en} ({branding.project_name_zh}) runner"
    )
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--question", required=True)
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--visit-id", default=None)
    parser.add_argument("--visit-turn", action="store_true")
    args = parser.parse_args()

    orchestrator = LangChainOrchestrator()
    if args.visit_turn:
        result = orchestrator.run_visit_turn(
            user_id=args.user_id,
            user_text=args.question,
            visit_id=args.visit_id,
            age=args.age,
            sex=args.sex,
            image_path=args.image,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    answer = orchestrator.run(
        user_id=args.user_id,
        user_text=args.question,
        age=args.age,
        sex=args.sex,
        image_path=args.image,
    )
    print(answer)


if __name__ == "__main__":
    main()
