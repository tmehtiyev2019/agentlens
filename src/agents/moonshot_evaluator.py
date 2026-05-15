"""
Moonshot Gate — first specialist agent in the AgentLens pipeline.

Answers three pass/fail questions about a product idea:
  Q1: Is this a real problem affecting millions of people?
  Q2: Is there a feasible, radical (10x improvement) solution?
  Q3: Is the enabling technology at TRL 4+ or achievable within 5 years?

passes_moonshot_gate is True only when all three answers are True.
"""

import os
import logging
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class MoonshotEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    problem_is_real: bool
    problem_explanation: str
    problem_evidence: list[str]

    solution_is_feasible: bool
    solution_explanation: str
    solution_evidence: list[str]

    technology_is_available: bool
    technology_trl: int = Field(ge=1, le=9)
    technology_explanation: str
    technology_evidence: list[str]

    passes_moonshot_gate: bool
    gate_failure_reason: str | None

    assumption_list: list[str]
    confidence_level: Literal["low", "medium", "high"]
    evidence_citations: list[str]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GATHER_PROMPT = """\
You are a moonshot research analyst. Your task is to gather hard evidence about an early-stage \
product idea before it is evaluated against the moonshot gate criteria.

Use the available search and academic tools to collect evidence for exactly these three questions:

Q1 — PROBLEM REALITY: Is this a real problem affecting millions of people? \
Search for documented market research, WHO/government data, academic studies, or news \
reporting that confirms the scale and severity of the problem.

Q2 — SOLUTION FEASIBILITY: Is there a feasible, radical (10x improvement) solution — \
not an incremental one? Search for prior art, patents, research prototypes, or startups \
attempting comparable solutions. Assess whether the proposed approach is radically better \
than the status quo.

Q3 — TECHNOLOGY READINESS: Is the enabling technology at TRL 4 or higher \
(demonstrated in lab environment) or realistically achievable within 5 years? \
Search Arxiv for relevant papers reporting experimental demonstrations. \
Use Wikipedia for TRL definitions and technology maturity context.

Conduct up to 9 tool calls total (~3 per question). Synthesize what you find into a \
thorough evidence summary. Be specific: cite paper titles, dataset names, statistics, \
and sources. Do not speculate beyond what the evidence shows.\
"""

SYNTHESIZE_PROMPT = """\
You are a moonshot gate evaluator. You have been given a product idea and a body of \
evidence gathered from web search and academic sources. Your task is to evaluate the \
idea against the three moonshot gate criteria and produce a structured verdict.

GATE CRITERIA — evaluate each as a strict boolean:

Q1 — PROBLEM REALITY
"Is this a real problem affecting millions of people with documented evidence?"
- True only if credible sources confirm scale (millions affected) and severity.
- False if the problem is niche, anecdotal, or undocumented.

Q2 — SOLUTION FEASIBILITY
"Is there a feasible, radical (10x improvement) solution — not incremental?"
- True only if the proposed approach represents a step-change improvement, not a 10–20% gain.
- False if the solution is a minor optimization of existing approaches.

Q3 — TECHNOLOGY READINESS
"Is the enabling technology at TRL 4+ (demonstrated in lab) or achievable within 5 years?"
- True if relevant components have been experimentally demonstrated (TRL 4–6) or if the \
  scientific consensus supports feasibility within a 5-year horizon.
- False if the technology requires breakthroughs with no documented pathway.

GATE RULE: passes_moonshot_gate = True ONLY IF ALL THREE criteria are True.
If any criterion is False, set passes_moonshot_gate = False and populate gate_failure_reason \
with a concise explanation of which criterion failed and why.

ADDITIONAL RULES:
- Never assert regulatory compliance or guarantee market success.
- Populate assumption_list with all significant assumptions made in your assessment.
- confidence_level reflects how much high-quality evidence supports your verdict: \
  "high" = multiple credible primary sources, "medium" = some indirect evidence, \
  "low" = sparse or conflicting evidence.
- evidence_citations must list every source (URL, paper title, dataset name) \
  that materially influenced your verdict.\
"""


# ---------------------------------------------------------------------------
# Agent run function
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    idea: str = state["idea"]
    log = logger.bind(agent="moonshot_evaluator", idea_snippet=idea[:80])

    last_exception: Exception | None = None

    for attempt in range(3):
        try:
            evidence_summary = _gather_evidence(idea, log)
            evaluation = _synthesize(idea, evidence_summary, log)
            log.info(
                "moonshot_gate_verdict",
                passes=evaluation.passes_moonshot_gate,
                confidence=evaluation.confidence_level,
                trl=evaluation.technology_trl,
            )
            return {"moonshot_evaluation": evaluation}

        except Exception as exc:
            last_exception = exc
            log.warning("moonshot_evaluator_retry", attempt=attempt + 1, error=str(exc))

    log.error("moonshot_evaluator_failed", error=str(last_exception))
    return {
        "moonshot_evaluation": None,
        "errors": [f"moonshot_evaluator: {last_exception}"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gather_evidence(idea: str, log) -> str:
    """Run the agentic tool-use loop to collect evidence for all three gate questions."""
    tools = [DuckDuckGoSearchRun(), ArxivQueryRun(), WikipediaQueryRun()]
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages: list = [
        SystemMessage(content=GATHER_PROMPT),
        HumanMessage(content=idea),
    ]

    for call_index in range(9):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            log.debug("evidence_loop_stopped", calls_made=call_index)
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            try:
                result = tool_map[tool_name].invoke(tc["args"])
                content = str(result)[:2000]
            except Exception as exc:
                # Surface tool errors as text so the LLM can reason around them
                content = f"[tool_error: {exc}]"
            messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    # The last AIMessage content is the evidence summary the LLM wrote after tool use
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content

    return "No evidence summary produced by the tool-use loop."


def _synthesize(idea: str, evidence_summary: str, log) -> MoonshotEvaluation:
    """Convert raw evidence into a typed MoonshotEvaluation via structured output."""
    llm = ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    llm_structured = llm.with_structured_output(MoonshotEvaluation)

    synthesis_input = (
        f"IDEA:\n{idea}\n\n"
        f"GATHERED EVIDENCE:\n{evidence_summary}"
    )

    log.debug("moonshot_synthesis_start")
    evaluation: MoonshotEvaluation = llm_structured.invoke(
        [
            SystemMessage(content=SYNTHESIZE_PROMPT),
            HumanMessage(content=synthesis_input),
        ]
    )
    return evaluation
