"""
LangGraph StateGraph orchestrator for the Techno-Economic Analysis pipeline.

Execution order:
  intent_classifier → moonshot_evaluator
    → FAIL: rejection_report (terminal)
    → PASS: [technical, market, risk, cost, rag] (parallel via Send)
          → kill_shot → END
"""

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agents.moonshot_evaluator import run as moonshot_evaluator_run, MoonshotEvaluation
from src.agents.technical_agent import run as technical_run, TechnicalAssessment
from src.agents.market_agent import run as market_run, MarketAssessment
from src.agents.risk_agent import run as risk_run, RiskAssessment
from src.agents.cost_estimation_agent import run as cost_run, CostEstimation
from src.agents.rag_agent import run as rag_run, RAGContext
from src.agents.kill_shot_agent import run as kill_shot_run, KillShotExperiment
from src.guardrails.intent_classifier import classify_intent


# ---------------------------------------------------------------------------
# Shared state
#
# TypedDict with Annotated reducers is the LangGraph pattern for shared state.
# Each field is Optional because not every node writes to every field —
# the moonshot gate may terminate early, or a parallel agent may fail.
#
# Reducer rules:
#   - errors: operator.add  →  parallel agents each append without overwriting
#   - all other fields: default (last-write-wins, one writer per field)
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Input
    idea: str

    # Guardrail
    intent_safe: bool | None

    # Moonshot gate
    moonshot_evaluation: MoonshotEvaluation | None

    # Parallel specialist outputs (each written by exactly one agent)
    technical_output: TechnicalAssessment | None
    market_output: MarketAssessment | None
    risk_output: RiskAssessment | None
    cost_output: CostEstimation | None
    rag_output: RAGContext | None

    # Final sequential agent
    kill_shot: KillShotExperiment | None

    # Accumulated errors from any node; operator.add merges lists from parallel writes
    errors: Annotated[list[str], operator.add]

    # HITL fields — populated when graph pauses after parallel agents
    human_decision: str | None          # "approved" | "rejected"
    human_comment:  str | None          # injected into kill shot prompt if provided

    # Tracks whether RAG retrieval found grounding context
    grounded: bool


# ---------------------------------------------------------------------------
# Node functions — thin wrappers so each node name is explicit in traces
# ---------------------------------------------------------------------------

def intent_classifier_node(state: GraphState) -> dict:
    safe, reason = classify_intent(state["idea"])
    if not safe:
        return {"intent_safe": False, "errors": [f"Intent blocked: {reason}"]}
    return {"intent_safe": True}


def moonshot_evaluator_node(state: GraphState) -> dict:
    return moonshot_evaluator_run(state)


def technical_node(state: GraphState) -> dict:
    return technical_run(state)


def market_node(state: GraphState) -> dict:
    return market_run(state)


def risk_node(state: GraphState) -> dict:
    return risk_run(state)


def cost_node(state: GraphState) -> dict:
    return cost_run(state)


def rag_node(state: GraphState) -> dict:
    return rag_run(state)


def human_review_node(state: GraphState) -> dict:
    # Intentionally empty: this node exists solely as the interrupt_after target.
    # LangGraph pauses here after all 5 parallel agents complete; the API layer
    # calls graph.update_state() with human_decision + human_comment, then resumes.
    return {}


def kill_shot_node(state: GraphState) -> dict:
    return kill_shot_run(state)


def parallel_spawn_node(state: GraphState) -> dict:
    # Pass-through node; direct edges to all 5 parallel agents trigger them simultaneously.
    return {}


def rejection_report_node(state: GraphState) -> dict:
    """Terminal node when moonshot gate fails. State already has the explanation."""
    return {}


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: GraphState) -> str:
    if not state.get("intent_safe"):
        return "rejection_report"
    return "moonshot_evaluator"


def route_after_moonshot(state: GraphState) -> str:
    evaluation: MoonshotEvaluation | None = state.get("moonshot_evaluation")
    if evaluation and evaluation.passes_moonshot_gate:
        return "parallel_spawn"
    return "rejection_report"


def route_after_human_review(state: GraphState) -> str:
    if state.get("human_decision") == "rejected":
        return "rejection_report"
    return "kill_shot_agent"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("moonshot_evaluator", moonshot_evaluator_node)
    graph.add_node("technical", technical_node)
    graph.add_node("market", market_node)
    graph.add_node("risk", risk_node)
    graph.add_node("cost", cost_node)
    graph.add_node("rag", rag_node)
    graph.add_node("parallel_spawn", parallel_spawn_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("kill_shot_agent", kill_shot_node)
    graph.add_node("rejection_report", rejection_report_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    # Intent classifier → conditional route
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "moonshot_evaluator": "moonshot_evaluator",
            "rejection_report": "rejection_report",
        },
    )

    # Moonshot gate → conditional route
    graph.add_conditional_edges(
        "moonshot_evaluator",
        route_after_moonshot,
        {
            "parallel_spawn": "parallel_spawn",
            "rejection_report": "rejection_report",
        },
    )

    # parallel_spawn fans out to all 5 agents simultaneously
    for node in ["technical", "market", "risk", "cost", "rag"]:
        graph.add_edge("parallel_spawn", node)

    # All parallel agents converge on human_review (sync point before HITL pause)
    for node in ["technical", "market", "risk", "cost", "rag"]:
        graph.add_edge(node, "human_review")

    # human_review → conditional: approved → kill_shot, rejected → rejection_report
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "kill_shot_agent": "kill_shot_agent",
            "rejection_report": "rejection_report",
        },
    )

    # Terminal edges
    graph.add_edge("kill_shot_agent", END)
    graph.add_edge("rejection_report", END)

    # interrupt_after pauses the graph at human_review for HITL input
    return graph.compile(interrupt_after=["human_review"], checkpointer=checkpointer)


# Compiled graph without checkpointer — used only by the benchmark runner
tea_graph = build_graph()
