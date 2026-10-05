from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ── State ──────────────────────────────────────────
class AgentState(TypedDict):
    user_query: str
    query_type: str
    retrieved_context: str
    agent_response: str
    evaluation_result: str
    final_response: str
    messages: List[dict]


# ── Graph Builder ───────────────────────────────────
def create_graph(router, rag, task, general, evaluator):

    # Node functions
    def router_node(state):
        return router.run(state)

    def rag_node(state):
        return rag.run(state)

    def task_node(state):
        return task.run(state)

    def general_node(state):
        return general.run(state)

    def evaluator_node(state):
        return evaluator.run(state)

    # Conditional edge function
    def route_decision(state):
        qt = state.get("query_type", "general")
        if qt == "rag":
            return "rag"
        elif qt == "task":
            return "task"
        else:
            return "general"

    # Evaluator decision
    def eval_decision(state):
        if state.get("evaluation_result") == "pass":
            return "pass"
        return "fail"

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router",    router_node)
    graph.add_node("rag",       rag_node)
    graph.add_node("task",      task_node)
    graph.add_node("general",   general_node)
    graph.add_node("evaluator", evaluator_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edge — Router se 3 agents
    graph.add_conditional_edges("router", route_decision, {
        "rag":     "rag",
        "task":    "task",
        "general": "general"
    })

    # Sab agents evaluator pe jayenge
    graph.add_edge("rag",     "evaluator")
    graph.add_edge("task",    "evaluator")
    graph.add_edge("general", "evaluator")

    # Evaluator se end
    graph.add_conditional_edges("evaluator", eval_decision, {
        "pass": END,
        "fail": END
    })

    return graph.compile()