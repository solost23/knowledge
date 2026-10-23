import re


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def score_agent_rows(rows: list[dict]) -> dict:
    total = len(rows)
    agent_rows = [row for row in rows if row.get("expected_mode") == "agent"]
    route_hits = [row.get("mode") == row.get("expected_mode") for row in rows]

    answer_recalls = []
    plan_recalls = []
    expected_tool_hits = []
    plan_valid = []
    planned_steps = 0
    completed_steps = 0
    agent_successes = 0
    for row in rows:
        terms = row.get("answer_terms", [])
        if terms:
            answer = _normalized(row.get("answer", ""))
            answer_recalls.append(sum(_normalized(term) in answer for term in terms) / len(terms))

    for row in agent_rows:
        plan = row.get("plan", [])
        steps = row.get("steps", [])
        min_steps = row.get("min_plan_steps", 1)
        valid = 1 <= len(plan) <= 5 and len(plan) >= min_steps
        plan_valid.append(valid)
        planned_steps += len(plan)
        completed_steps += len(steps)
        agent_successes += bool(row.get("success")) and row.get("mode") == "agent"

        plan_terms = row.get("plan_terms", [])
        if plan_terms:
            normalized_plan = _normalized(" ".join(plan))
            plan_recalls.append(
                sum(_normalized(term) in normalized_plan for term in plan_terms) / len(plan_terms)
            )

        expected_tools = row.get("expected_tools", [])
        if expected_tools:
            used_tools = set(row.get("tools", []))
            expected_tool_hits.extend(tool in used_tools for tool in expected_tools)

    return {
        "cases": total,
        "route_accuracy": round(sum(route_hits) / max(total, 1), 4),
        "agent_cases": len(agent_rows),
        "agent_completion_rate": round(agent_successes / max(len(agent_rows), 1), 4),
        "agent_plan_valid_rate": round(sum(plan_valid) / max(len(plan_valid), 1), 4),
        "agent_step_completion_rate": round(completed_steps / max(planned_steps, 1), 4),
        "agent_plan_term_recall": round(sum(plan_recalls) / max(len(plan_recalls), 1), 4),
        "expected_tool_recall": round(
            sum(expected_tool_hits) / max(len(expected_tool_hits), 1), 4
        ),
        "answer_term_recall": round(
            sum(answer_recalls) / max(len(answer_recalls), 1), 4
        ),
    }
