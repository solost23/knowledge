#!/usr/bin/env python
"""End-to-end Agent routing and execution evaluation.

Run with: uv run --locked python eval/agent.py [--limit 4]
The evaluation documents must already be uploaded to the configured knowledge base.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from eval.agent_metrics import score_agent_rows
from services.ask import AskService


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)[:64] or "latest"


def _render_report(summary: dict, rows: list[dict]) -> str:
    lines = ["# Agent 评测报告", "", "| 指标 | 值 |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in summary.items())
    lines.extend(["", "## 逐题结果", "", "| ID | 路由 | 预期 | 状态 | 计划步数 | 工具 | 关键词召回 |", "|---|---|---|---:|---:|---|---:|"])
    for row in rows:
        lines.append(
            f"| {row['id']} | {row.get('mode', '-')} | {row['expected_mode']} | "
            f"{row['http_status']} | {len(row.get('plan', []))} | "
            f"{', '.join(row.get('tools', [])) or '-'} | {row.get('answer_term_recall', '-')} |"
        )
    lines.extend(["", "## 未通过用例", ""])
    failures = [row for row in rows if not row.get("case_pass")]
    if failures:
        lines.extend(
            f"- {row['id']}: actual={row.get('mode')}, expected={row['expected_mode']}, "
            f"answer_term_recall={row.get('answer_term_recall')}"
            for row in failures
        )
    else:
        lines.append("无")
    return "\n".join(lines) + "\n"


def run(questions: list[dict], service=None) -> tuple[dict, list[dict]]:
    service = service or AskService()
    rows = []
    for case in questions:
        result, status = service.ask(case["task"])
        data = result.get("data") or {}
        answer = data.get("answer", "")
        terms = case.get("answer_terms", [])
        normalized_answer = re.sub(r"\s+", "", answer).lower()
        term_hits = [re.sub(r"\s+", "", term).lower() in normalized_answer for term in terms]
        steps = data.get("steps", [])
        tools = sorted({name for step in steps for name in step.get("tools", [])})
        rows.append({
            **case,
            "mode": data.get("mode"),
            "route_reason": data.get("route_reason"),
            "success": status < 400 and result.get("success", False),
            "http_status": status,
            "message": result.get("message", ""),
            "answer": answer,
            "plan": data.get("plan", []),
            "steps": steps,
            "tools": tools,
            "answer_term_recall": round(sum(term_hits) / len(term_hits), 4) if term_hits else None,
        })
        passed = (
            rows[-1]["mode"] == case["expected_mode"]
            and rows[-1]["success"]
            and rows[-1]["answer_term_recall"] in (None, 1.0)
        )
        if case["expected_mode"] == "agent":
            plan = rows[-1]["plan"]
            expected_tools = set(case.get("expected_tools", []))
            plan_text = re.sub(r"\s+", "", "".join(plan)).lower()
            passed = (
                passed
                and len(plan) >= case.get("min_plan_steps", 1)
                and len(plan) <= 5
                and all(re.sub(r"\s+", "", term).lower() in plan_text for term in case.get("plan_terms", []))
                and expected_tools.issubset(set(rows[-1]["tools"]))
            )
        rows[-1]["case_pass"] = passed
        label = "PASS" if passed else "FAIL"
        print(f"[{label}] {case['id']} expected={case['expected_mode']} actual={rows[-1]['mode']}")
    summary = score_agent_rows(rows)
    summary["passed_cases"] = sum(row["case_pass"] for row in rows)
    summary["case_pass_rate"] = round(summary["passed_cases"] / max(len(rows), 1), 4)
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=os.path.join(ROOT, "eval", "agent_questions.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    summary, rows = run(questions)
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    tag = _safe_tag(args.tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    report_dir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f"agent_{tag}.json")
    markdown_path = os.path.join(report_dir, f"agent_{tag}.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "rows": rows}, file, ensure_ascii=False, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(_render_report(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"reports: {json_path}, {markdown_path}")


if __name__ == "__main__":
    main()
