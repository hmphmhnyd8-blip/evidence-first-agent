import traceback

from langchain_core.tools import tool

from evidence_graph import build_evidence_graph


def make_evidence_search(model):
    """将有界证据检查子图封装为 Agent 工具。"""

    graph = build_evidence_graph(model, max_llm_calls=3)

    @tool
    def search_documents(question: str) -> dict:
        """检索本地项目规则和 AIS 资料，并检查证据是否充分。

        用于平均航速计算规则、缺失值处理、AIS 字段等资料问题。
        question 应是包含必要上下文的完整问题。
        最多改写检索一次；不查询具体船舶的数值统计。

        返回状态：
        ok：证据充分；
        insufficient：检查完成，但证据不足；
        tool_error：执行失败或达到预算，检查未完成。
        """
        question = question.strip()

        if not question:
            return {
                "status": "tool_error",
                "error_type": "empty_question",
                "message": "检索问题不能为空。",
            }

        try:
            state = graph.invoke({
                "question": question,
                "query": question,
                "matches": [],
                "decision": {},
                "rewrite_count": 0,
                "llm_calls": 0,
            })

        except Exception as exc:
            traceback.print_exc()

            return {
                "status": "tool_error",
                "error_type": type(exc).__name__,
                "message": (
                    "资料检查未完成，请根据异常类型排查。"
                    "不能解释为资料不存在，也不要反复调用同一失败请求。"
                ),
            }

        decision = state["decision"]

        return {
            "status": (
                "ok" if decision["sufficient"] else "insufficient"
            ),
            "question": question,
            "decision": decision,
            "matches": state["matches"],
            "count": len(state["matches"]),
            "rewrite_count": state["rewrite_count"],
            "llm_calls": state["llm_calls"],
        }

    return search_documents