import argparse

from medagent.config import AppConfig
from medagent.orchestrator import Orchestrator


def main() -> None:
    branding = AppConfig().branding
    parser = argparse.ArgumentParser(
        description=f"{branding.project_name_en} ({branding.project_name_zh}) runner"
    )
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--question", required=True)
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", default=None)
    args = parser.parse_args()

    orchestrator = Orchestrator()
    answer = orchestrator.run(
        user_id=args.user_id,
        user_text=args.question,
        age=args.age,
        sex=args.sex,
    )
    print(answer)


if __name__ == "__main__":
    main()
