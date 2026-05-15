"""
LangGraph StateGraph orchestrator for the Techno-Economic Analysis pipeline.

Execution order:
  intent_classifier
    → CHAT:  chat_response (terminal)
    → FAIL:  rejection_report (terminal)
    → PASS:  moonshot_evaluator
               → FAIL: rejection_report (terminal)
               → PASS: [technical, market, risk, cost, rag] (parallel via Send)
                         → human_review (HITL pause)
                           → REJECTED: rejection_report (terminal)
                           → REVISE:   parallel_spawn (loop back with critique)
                           → APPROVED: kill_shot_agent → END

A human reviewer can revise the analysis up to MAX_REVISIONS times. The
critique text is stored in state["human_critique"] and read by each specialist
agent on the next round. Beyond MAX_REVISIONS, revise is treated as approve to
prevent runaway loops.
"""

MAX_REVISIONS = 2

import operator
import os
from typing import Annotated, Any, Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from src.agents.moonshot_evaluator import run as moonshot_evaluator_run, MoonshotEvaluation
from src.agents.technical_agent import run as technical_run, TechnicalAssessment
from src.agents.market_agent import run as market_run, MarketAssessment
from src.agents.risk_agent import run as risk_run, RiskAssessment
from src.agents.cost_estimation_agent import run as cost_run, CostEstimation
from src.agents.rag_agent import run as rag_run, RAGContext
from src.agents.kill_shot_agent import run as kill_shot_run, KillShotExperiment
from src.guardrails.intent_classifier import classify_intent

logger = structlog.get_logger()

_CHAT_SYSTEM = """\
You are AgentLens — a techno-economic analysis system that evaluates moonshot product ideas
using a LangGraph pipeline of specialist AI agents (technical feasibility, market opportunity,
risk assessment, cost estimation, and knowledge retrieval).

The user sent a conversational message rather than a moonshot idea to evaluate.
Respond naturally in 3-5 sentences:
1. Acknowledge their message warmly
2. Explain in one sentence what AgentLens does
3. Invite them to describe any product, technology, or research idea — even rough ones

Be friendly and brief. Do NOT evaluate their message. End with an open invitation.
Examples of ideas they could submit (mention 1-2): solid-state batteries, ocean plastic
collection, autonomous surgical robots, nuclear fusion power plants, brain-computer interfaces.\
"""


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Input
    idea: str

    # Guardrail + routing
    intent_safe: bool | None
    conversation_type: Literal["idea", "chat"] | None

    # Conversational response (populated when conversation_type == "chat")
    chat_response: str | None

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

    # Accumulated errors from any node
    errors: Annotated[list[str], operator.add]

    # HITL fields
    human_decision: str | None       # "approved" | "rejected" | "revise"
    human_comment: str | None
    human_critique: str | None       # latest revision request — read by specialist agents
    revision_count: int              # number of revise-loops completed; capped at MAX_REVISIONS

    # Tracks whether RAG retrieval found grounding context
    grounded: bool


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def intent_classifier_node(state: GraphState) -> dict:
    safe, reason, input_type = classify_intent(state["idea"])
    if not safe:
        return {
            "intent_safe": False,
            "conversation_type": "idea",
            "errors": [f"Intent blocked: {reason}"],
        }
    return {"intent_safe": True, "conversation_type": input_type}


def chat_response_node(state: GraphState) -> dict:
    """Generates a friendly conversational reply when the input is not a moonshot idea."""
    log = logger.bind(agent="chat_response_node")
    log.info("generating chat response")
    try:
        llm = ChatOpenAI(model=os.getenv("AGENT_MODEL", "gpt-4o-mini"), temperature=0.4)
        response = llm.invoke(
            [SystemMessage(content=_CHAT_SYSTEM), HumanMessage(content=state["idea"])]
        )
        return {"chat_response": str(response.content)}
    except Exception as exc:
        log.exception("chat_response_node failed", error=str(exc))
        return {
            "chat_response": (
                "Hi! I'm AgentLens — I analyze moonshot product ideas using a panel of "
                "specialist AI agents. Describe a technology or product concept and I'll "
                "give you a full techno-economic assessment. What idea would you like to explore?"
            )
        }


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
    return {}


def kill_shot_node(state: GraphState) -> dict:
    return kill_shot_run(state)


def parallel_spawn_node(state: GraphState) -> dict:
    # Pass-through; direct edges to all 5 parallel agents trigger them simultaneously.
    return {}


def rejection_report_node(state: GraphState) -> dict:
    """Terminal node when moonshot gate or HITL rejects. State already has the explanation."""
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_intent(state: GraphState) -> str:
    if not state.get("intent_safe"):
        return "rejection_report"
    if state.get("conversation_type") == "chat":
        return "chat_responder"
    return "moonshot_evaluator"


def route_after_moonshot(state: GraphState) -> str:
    evaluation: MoonshotEvaluation | None = state.get("moonshot_evaluation")
    if evaluation and evaluation.passes_moonshot_gate:
        return "parallel_spawn"
    return "rejection_report"


def route_after_human_review(state: GraphState) -> str:
    decision = state.get("human_decision")
    if decision == "rejected":
        return "rejection_report"
    if decision == "revise":
        # Cap revisions to avoid runaway loops — beyond cap, proceed to kill_shot.
        if (state.get("revision_count") or 0) >= MAX_REVISIONS:
            logger.warning(
                "max_revisions_reached_forcing_proceed",
                revision_count=state.get("revision_count"),
                max_revisions=MAX_REVISIONS,
            )
            return "kill_shot_agent"
        return "parallel_spawn"
    return "kill_shot_agent"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> Any:
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("chat_responder", chat_response_node)
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
            "chat_responder": "chat_responder",
            "moonshot_evaluator": "moonshot_evaluator",
            "rejection_report": "rejection_report",
        },
    )

    # Chat path → terminal
    graph.add_edge("chat_responder", END)

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

    # All parallel agents converge on human_review
    for node in ["technical", "market", "risk", "cost", "rag"]:
        graph.add_edge(node, "human_review")

    # human_review → conditional: approved → kill_shot, rejected → rejection_report,
    # revise → parallel_spawn (loop back so specialists re-run with the critique).
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "kill_shot_agent": "kill_shot_agent",
            "rejection_report": "rejection_report",
            "parallel_spawn": "parallel_spawn",
        },
    )

    # Terminal edges
    graph.add_edge("kill_shot_agent", END)
    graph.add_edge("rejection_report", END)

    return graph.compile(interrupt_after=["human_review"], checkpointer=checkpointer)


# Compiled graph without checkpointer — used only by the benchmark runner
tea_graph = build_graph()
