"""Shared validation for agent requests.

Keeping these rules in one module prevents the synchronous and background
execution paths from drifting apart.
"""

from uuid import UUID

from app.agents.state import MAX_STEP_LENGTH, MAX_TASK_LENGTH, InvalidAgentRequest


def validate_execution_id(execution_id: str) -> str:
    try:
        return str(UUID(str(execution_id)))
    except (ValueError, TypeError, AttributeError) as error:
        raise InvalidAgentRequest("execution_id 必须是有效的 UUID") from error


def validate_task(task: str) -> str:
    normalized = task.strip() if isinstance(task, str) else ""
    if not normalized:
        raise InvalidAgentRequest("任务内容不能为空")
    if len(normalized) > MAX_TASK_LENGTH:
        raise InvalidAgentRequest(f"任务内容不能超过 {MAX_TASK_LENGTH} 个字符")
    return normalized


def validate_plan(plan: list[str] | None) -> list[str]:
    if plan is None:
        return []
    if not isinstance(plan, list):
        raise InvalidAgentRequest("执行计划必须是步骤列表")

    steps: list[str] = []
    for step in plan:
        if not isinstance(step, str) or not step.strip():
            raise InvalidAgentRequest("执行步骤不能为空")
        normalized = step.strip()
        if len(normalized) > MAX_STEP_LENGTH:
            raise InvalidAgentRequest(f"单个执行步骤不能超过 {MAX_STEP_LENGTH} 个字符")
        steps.append(normalized)
    return steps
