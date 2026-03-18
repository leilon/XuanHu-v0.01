from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from medagent.langgraph_orchestrator import LangChainOrchestrator


DEMO_CASES = [
    {
        "title": "呼吸道发热场景",
        "user_id": "demo_case_01",
        "question": "我发烧两天了，还咳嗽，今天有点喘，需要怎么处理？",
        "age": 29,
        "sex": "female",
    },
    {
        "title": "胸痛高风险场景",
        "user_id": "demo_case_02",
        "question": "我是男的，胸口压着疼，还出汗，刚才差点喘不上气，这种情况严重吗？",
        "age": 56,
        "sex": "male",
    },
    {
        "title": "儿科发热抽搐场景",
        "user_id": "demo_case_03",
        "question": "孩子发烧两天了，刚才抽了两分钟，现在有点没精神，要不要马上去医院？",
        "age": 3,
        "sex": "female",
    },
    {
        "title": "育龄女性腹痛场景",
        "user_id": "demo_case_04",
        "question": "我下腹痛两天了，月经也推迟了，还有点恶心，这种情况要注意什么？",
        "age": 26,
        "sex": "female",
    },
    {
        "title": "皮疹过敏场景",
        "user_id": "demo_case_05",
        "question": "我昨晚吃完海鲜以后身上起了很多红疹，很痒，要怎么处理？",
        "age": 23,
        "sex": "male",
    },
    {
        "title": "神经系统高风险场景",
        "user_id": "demo_case_06",
        "question": "我今天突然头痛得特别厉害，还吐了两次，左手有点没劲，这种情况怎么办？",
        "age": 61,
        "sex": "male",
    },
    {
        "title": "医学科普场景",
        "user_id": "demo_case_07",
        "question": "支原体肺炎到底是什么，会不会传染给家里人？",
        "age": 30,
        "sex": "female",
    },
    {
        "title": "化验单解读场景",
        "user_id": "demo_case_08",
        "question": "我这张化验单写白细胞12.8，C反应蛋白48，这是什么意思？需要怎么处理？",
        "age": 34,
        "sex": "male",
    },
    {
        "title": "慢病用户首次问诊",
        "user_id": "demo_case_09",
        "question": "我有糖尿病，一直吃二甲双胍，这两天发烧咳嗽，应该注意什么？",
        "age": 45,
        "sex": "male",
    },
    {
        "title": "慢病用户复问用药",
        "user_id": "demo_case_09",
        "question": "接着刚才那个情况，我现在头痛发热，能自己吃布洛芬吗？",
        "age": 45,
        "sex": "male",
    },
]


def build_report(output_path: Path) -> None:
    orchestrator = LangChainOrchestrator()
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# QingNang-ClinicOS Demo 报告")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 基模：{orchestrator.config.default_profile.name}")
    lines.append(f"- 样例数：{len(DEMO_CASES)}")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 本报告基于 AutoDL 实机推理结果生成。")
    lines.append("- 所有样例均通过同一个 `HuatuoGPT-Vision-7B-Qwen2.5VL` 基模完成。")
    lines.append("- 多 agent 通过 prompt profiles 实现，未使用单独 adapter。")
    lines.append("")

    for idx, case in enumerate(DEMO_CASES, start=1):
        route = orchestrator.intent_router.route(case["question"])
        start = time.perf_counter()
        try:
            answer = orchestrator.run(
                user_id=case["user_id"],
                user_text=case["question"],
                age=case.get("age"),
                sex=case.get("sex"),
                image_path=case.get("image"),
            )
            elapsed = time.perf_counter() - start
        except Exception as exc:  # pragma: no cover - runtime report fallback
            answer = f"[运行失败] {exc}"
            elapsed = time.perf_counter() - start

        lines.append(f"## 示例 {idx}：{case['title']}")
        lines.append("")
        lines.append(f"- 用户ID：`{case['user_id']}`")
        lines.append(f"- 预估意图：`{route.intent}`")
        lines.append(f"- 年龄：`{case.get('age')}`")
        lines.append(f"- 性别：`{case.get('sex')}`")
        lines.append(f"- 耗时：`{elapsed:.2f}s`")
        lines.append("")
        lines.append("**输入**")
        lines.append("")
        lines.append(f"> {case['question']}")
        lines.append("")
        lines.append("**输出**")
        lines.append("")
        lines.append("```text")
        lines.append(answer)
        lines.append("```")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 10-case demo report for QingNang-ClinicOS.")
    parser.add_argument(
        "--output",
        default="docs/reports/demo_10_cases_report_zh.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()
    build_report(Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
