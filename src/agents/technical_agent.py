"""
Technical feasibility agent.

Gathers evidence from Arxiv and Wikipedia via a tool-calling loop, then
synthesizes a structured TechnicalAssessment using with_structured_output.
Runs in parallel with market_agent and risk_agent — reads state["idea"],
writes state["technical_output"].
"""

import os
from typing import Literal

import structlog
from langchain_openai import ChatOpenAI
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TechnicalAssessment(BaseModel):
    trl_level: int = Field(
        ...,
        ge=1,
        le=9,
        description="Technology Readiness Level on the NASA 1-9 scale.",
    )
    trl_justification: str = Field(
        ...,
        description="One-paragraph justification for the assigned TRL score.",
    )
    key_blockers: list[str] = Field(
        ...,
        min_length=1,
        description="Top 3 concrete technical blockers preventing deployment today.",
    )
    required_breakthroughs: list[str] = Field(
        ...,
        description="Specific unsolved problems that must be invented or solved before the idea is viable.",
    )
    time_to_prototype_years: float = Field(
        ...,
        description="Realistic years to a working prototype under adequate funding.",
    )
    assumption_list: list[str] = Field(
        ...,
        description="Explicit assumptions baked into this assessment.",
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Agent's confidence in its own assessment given evidence quality.",
    )
    evidence_citations: list[str] = Field(
        ...,
        description="Arxiv paper IDs or Wikipedia article titles used as evidence.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a skeptical technology readiness analyst. Your job is to assess the \
technological readiness of an early-stage idea using the NASA Technology \
Readiness Level (TRL) scale:

  TRL 1-3: Basic research — principles observed, concept formulated, analytical proof
  TRL 4-6: Lab / prototype — validated in lab, validated in relevant environment
  TRL 7-9: Production ready — demonstrated in operational environment, proven system

Use the ArxivQueryRun tool to search for peer-reviewed papers that support or \
challenge the TRL claim. Use WikipediaQueryRun for background on the enabling \
technology domain. Run multiple searches — at least one Arxiv search and one \
Wikipedia search are expected.

Be skeptical. Founders and researchers consistently overestimate TRL by 2-3 \
levels. Challenge optimistic claims. If cutting-edge papers describe the \
technology as nascent or speculative, reflect that in a low TRL score. \
Identify specific unsolved problems (e.g., materials limitations, compute \
requirements, safety certification gaps) — do not list generic risks. \
Cite specific Arxiv paper IDs or Wikipedia article titles as evidence.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_tools = [ArxivQueryRun(), WikipediaQueryRun()]
_tool_map = {t.name: t for t in _tools}


def run(state: dict) -> dict:
    idea: str = state["idea"]

    log = logger.bind(agent="technical_agent", idea_preview=idea[:80])
    log.info("starting technical assessment")

    try:
        model_name = os.getenv("AGENT_MODEL", "gpt-4o-mini")

        llm_with_tools = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).bind_tools(_tools)

        messages: list = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=idea),
        ]

        # Tool-calling evidence gathering loop (max 6 rounds)
        for round_idx in range(6):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                log.info("tool loop complete", rounds=round_idx + 1)
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                if tool_name not in _tool_map:
                    log.warning("unknown tool called", tool=tool_name)
                    continue
                result = _tool_map[tool_name].invoke(tc["args"])
                # Truncate to 2 000 chars to stay within context budget
                messages.append(
                    ToolMessage(
                        content=str(result)[:2000],
                        tool_call_id=tc["id"],
                    )
                )

        # Structured synthesis — uses a separate LLM call without tools bound
        # so the model is forced to emit a schema-conformant response only.
        structured_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).with_structured_output(TechnicalAssessment)

        assessment: TechnicalAssessment = structured_llm.invoke(
            [
                *messages,
                HumanMessage(
                    content=(
                        "Based on the research above, produce a structured "
                        "TechnicalAssessment. Be concrete, cite evidence, and "
                        "do not inflate the TRL score."
                    )
                ),
            ]
        )

        log.info(
            "technical assessment complete",
            trl=assessment.trl_level,
            confidence=assessment.confidence_level,
        )
        return {"technical_output": assessment}

    except Exception as exc:  # noqa: BLE001
        log.exception("technical_agent failed", error=str(exc))
        return {
            "technical_output": None,
            "errors": [f"technical_agent: {exc}"],
        }
