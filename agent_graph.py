import json
import traceback

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from openai import APITimeoutError

# 复用已经写好的工具执行和日志函数
from tool_runtime import execute_tool, save_tool_log


class AgentState(MessagesState):
    """保存对话消息，以及当前用户问题的模型请求次数。"""

    llm_calls: int


def run_agent_turn(model_with_tools, messages, tools_by_name):
    """
    使用 LangGraph 处理一次用户输入。

    更新传入的 messages 列表，保留完整对话历史。
    打印工具执行过程和最终回复。
    """
    max_rounds = 6

    # 这些内部函数可以使用外部传入的模型和工具字典
    def model_node(state: AgentState) -> dict:
        """请求模型，返回新的模型消息。"""
        current_round = state["llm_calls"] + 1

        print(f"\n--- 第 {current_round} 轮模型请求 ---")

        try:
            response = model_with_tools.invoke(state["messages"])

        except APITimeoutError:
            response = AIMessage(
                content="模型请求超时，本轮未完成。请稍后重新输入问题。"
            )

        if response.tool_calls:
            print("工具调用请求：")
            print(json.dumps(
                response.tool_calls,
                ensure_ascii=False,
                indent=2,
            ))

        # 只返回新消息；历史消息由 LangGraph 合并保留
        return {
            "messages": [response],
            "llm_calls": current_round,
        }

    def tools_node(state: AgentState) -> dict:
        """执行最新模型回复中提出的所有工具调用。"""
        last_message = state["messages"][-1]

        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_result = execute_tool(
                tool_call=tool_call,
                tools_by_name=tools_by_name,
            )

            print("工具执行结果：")
            print(json.dumps(
                tool_result,
                ensure_ascii=False,
                indent=2,
            ))

            # 一个工具请求，对应一条工具结果消息
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                    tool_call_id=tool_call["id"],
                )
            )

            try:
                save_tool_log(
                    tool_call["name"],
                    tool_call["args"],
                    tool_result,
                )
            except OSError:
                traceback.print_exc()
                print("本次工具日志未能保存。")

        return {"messages": tool_messages}

    def route_after_model(state: AgentState) -> str:
        """模型回复后，决定执行工具还是结束。"""
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return "end"

    def route_after_tools(state: AgentState) -> str:
        """工具执行后，决定继续请求模型还是到达上限。"""
        if state["llm_calls"] >= max_rounds:
            return "limit"

        return "model"

    def limit_node(state: AgentState) -> dict:
        """达到请求次数上限时，记录未完成提示。"""
        notice = (
            "本次查询达到模型请求轮数上限，"
            "尚未完成，请缩小查询范围。"
        )

        return {
            "messages": [AIMessage(content=notice)],
        }

    # 构建流程
    builder = StateGraph(AgentState)

    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_node("limit", limit_node)

    builder.add_edge(START, "model")

    builder.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "model": "model",
            "limit": "limit",
        },
    )

    builder.add_edge("limit", END)

    graph = builder.compile()

    # 每次用户输入都重新计数，但携带已有对话历史
    final_state = graph.invoke({
        "messages": messages,
        "llm_calls": 0,
    })

    # 原地更新外部列表，使下一次用户输入能继续使用历史
    messages[:] = final_state["messages"]

    print("助手：", final_state["messages"][-1].content)