import json

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agents.state import (
    MAX_STEP_LENGTH,
    AgentState,
    TaskPlan,
)
from app.agents.tools import (
    extract_sources,
    list_knowledge_documents,
    message_text,
    search_knowledge,
)
from app.core.settings import config
from app.prompts.agent import (
    EXECUTOR_SYSTEM_PROMPT,
    EXECUTOR_TASK_PROMPT,
    FINALIZER_PROMPT,
    PLAN_PROMPT,
)


def compile_graph(
    planner, executor, finalizer, checkpointer=None, progress_callback=None
):
    def plan(state: AgentState):
        steps = [step.strip() for step in state["plan"] if step.strip()]
        if any(len(step) > MAX_STEP_LENGTH for step in steps):
            raise ValueError(f"每个执行步骤不能超过 {MAX_STEP_LENGTH} 个字符")
        if not steps and planner is not None:
            if progress_callback:
                progress_callback(
                    {
                        "stage": "planning",
                        "current_action": "正在拆分执行计划",
                    }
                )
            result = planner.invoke(PLAN_PROMPT.format_messages(task=state["task"]))
            steps = [step.strip() for step in result.steps if step.strip()]
        if not steps:
            raise ValueError("模型没有生成可执行步骤")
        if progress_callback:
            progress_callback(
                {
                    "stage": "executing",
                    "plan": steps,
                    "current_step": 0,
                    "active_step": None,
                    "current_action": "执行计划已生成，准备开始",
                }
            )
        return {"plan": steps, "current_step": 0}

    def execute_step(state: AgentState):
        index = state["current_step"]
        if progress_callback:
            progress_callback(
                {
                    "stage": "executing",
                    "plan": state["plan"],
                    "current_step": index,
                    "active_step": index + 1,
                    "current_action": f"正在执行第 {index + 1} 步：{state['plan'][index]}",
                }
            )
        prior_results = (
            "\n".join(
                f"{item['index']}. {item['task']}：{item['result']}"
                for item in state["steps"]
            )
            or "无"
        )
        prompt = EXECUTOR_TASK_PROMPT.format(
            task=state["task"],
            step_number=index + 1,
            total_steps=len(state["plan"]),
            step_task=state["plan"][index],
            prior_results=prior_results,
        )
        result = executor.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"recursion_limit": config.agent["recursion_limit"]},
        )
        messages = result["messages"]
        final_message = next(
            (
                message
                for message in reversed(messages)
                if not isinstance(message, ToolMessage)
            ),
            messages[-1],
        )
        tool_calls = list(
            dict.fromkeys(
                message.name
                for message in messages
                if isinstance(message, ToolMessage) and message.name
            )
        )
        step_result = {
            "index": index + 1,
            "task": state["plan"][index],
            "result": message_text(final_message.content),
            "tools": tool_calls,
        }
        if progress_callback:
            progress_callback(
                {
                    "stage": "executing",
                    "current_step": index + 1,
                    "active_step": None,
                    "current_action": f"第 {index + 1} 步已完成",
                    "steps": state["steps"] + [step_result],
                }
            )
        sources = state["sources"] + extract_sources(messages)
        unique_sources = []
        seen = set()
        for source in sources:
            key = (source["source"], source["content"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        return {"steps": state["steps"] + [step_result], "sources": unique_sources[:10]}

    def advance(state: AgentState):
        return {"current_step": state["current_step"] + 1}

    def finalize(state: AgentState):
        if progress_callback:
            progress_callback(
                {
                    "stage": "finalizing",
                    "current_step": len(state["plan"]),
                    "active_step": None,
                    "current_action": "正在整理步骤结果和参考来源",
                }
            )
        evidence = json.dumps(
            {"completed_steps": state["steps"], "sources": state["sources"]},
            ensure_ascii=False,
        )
        result = finalizer.invoke(
            FINALIZER_PROMPT.format_messages(task=state["task"], evidence=evidence)
        )
        return {"answer": message_text(result.content)}

    def route_after_advance(state: AgentState):
        return (
            "execute_step" if state["current_step"] < len(state["plan"]) else "finalize"
        )

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("advance", advance)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_step")
    graph.add_edge("execute_step", "advance")
    graph.add_conditional_edges("advance", route_after_advance)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def build_graph(llm, checkpointer=None, progress_callback=None):
    planner = llm.with_structured_output(TaskPlan, method="function_calling")
    executor = create_agent(
        llm,
        tools=[search_knowledge, list_knowledge_documents],
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
    )
    return compile_graph(
        planner,
        executor,
        llm,
        checkpointer=checkpointer,
        progress_callback=progress_callback,
    )
