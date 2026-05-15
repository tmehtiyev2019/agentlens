"""
Market opportunity agent.

Gathers evidence via DuckDuckGo web search and Wikipedia, then synthesizes a
structured MarketAssessment using with_structured_output. Runs in parallel
with technical_agent and risk_agent — reads state["idea"], writes
state["market_output"].
"""

import os
from typing import Literal

import structlog
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class MarketAssessment(BaseModel):
    tam_usd: float = Field(
        ...,
        description="Total Addressable Market in USD (global demand if 100% captured).",
    )
    sam_usd: float = Field(
        ...,
        description=(
            "Serviceable Addressable Market in USD (subset reachable with the "
            "proposed go-to-market model)."
        ),
    )
    som_usd: float = Field(
        ...,
        description=(
            "Serviceable Obtainable Market in USD — realistic 5-year capture. "
            "Should rarely exceed 5% of SAM for early-stage moonshots."
        ),
    )
    market_growth_rate_pct: float = Field(
        ...,
        description="Expected annual market growth rate as a percentage (e.g. 12.5 for 12.5%).",
    )
    top_competitors: list[str] = Field(
        ...,
        min_length=1,
        description="Top 3-5 competitors or substitute solutions with brief notes on each.",
    )
    competitive_moat: str = Field(
        ...,
        description=(
            "What makes this idea defensible — IP, network effects, switching costs, "
            "regulatory moat, etc. Must be specific, not generic."
        ),
    )
    time_to_market_years: float = Field(
        ...,
        description="Realistic years from today to first commercial revenue.",
    )
    assumption_list: list[str] = Field(
        ...,
        description="Explicit assumptions underlying the TAM/SAM/SOM calculation.",
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Agent's confidence in its own assessment given evidence quality.",
    )
    evidence_citations: list[str] = Field(
        ...,
        description="URLs, report names, or Wikipedia article titles used as evidence.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a rigorous market sizing analyst. Your job is to estimate the market \
opportunity for an early-stage idea using a bottoms-up approach wherever possible.

Definitions you must respect:
  TAM (Total Addressable Market): global revenue opportunity if you captured 100%
  SAM (Serviceable Addressable Market): the portion reachable with this go-to-market strategy
  SOM (Serviceable Obtainable Market): realistic 5-year revenue capture — almost always <5% of SAM \
for an early-stage company entering an established market, and <10% even for new-market creation

Use DuckDuckGoSearchRun to find market research reports, industry analyst estimates, \
funding announcements, and competitor revenue figures. Use WikipediaQueryRun for \
background on the industry vertical. Run at least two DuckDuckGo searches and one \
Wikipedia lookup before synthesizing.

Be conservative. Moonshots often target markets that do not yet exist or are \
fragmented across incumbents. Explicitly flag whether this is new-market creation \
(demand must be educated) versus displacement (demand exists, switching cost is the \
barrier). Identify the top 3-5 competitors or substitutes and their specific weaknesses. \
Cite your sources.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    idea: str = state["idea"]

    log = logger.bind(agent="market_agent", idea_preview=idea[:80])
    log.info("starting market assessment")

    tools = [TavilySearchResults(max_results=5), WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
    tool_map = {t.name: t for t in tools}

    try:
        model_name = os.getenv("AGENT_MODEL", "gpt-4o-mini")

        llm_with_tools = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).bind_tools(tools)

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
                if tool_name not in tool_map:
                    log.warning("unknown tool called", tool=tool_name)
                    continue
                result = tool_map[tool_name].invoke(tc["args"])
                # Truncate to 2 000 chars to stay within context budget
                messages.append(
                    ToolMessage(
                        content=str(result)[:2000],
                        tool_call_id=tc["id"],
                    )
                )

        # Structured synthesis — bind no tools so the model must emit the schema.
        structured_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).with_structured_output(MarketAssessment)

        assessment: MarketAssessment = structured_llm.invoke(
            [
                *messages,
                HumanMessage(
                    content=(
                        "Based on the research above, produce a structured "
                        "MarketAssessment. Be explicit about every assumption. "
                        "SOM must be conservative — justify why it is not higher."
                    )
                ),
            ]
        )

        log.info(
            "market assessment complete",
            tam=assessment.tam_usd,
            som=assessment.som_usd,
            confidence=assessment.confidence_level,
        )
        return {"market_output": assessment}

    except Exception as exc:  # noqa: BLE001
        log.exception("market_agent failed", error=str(exc))
        return {
            "market_output": None,
            "errors": [f"market_agent: {exc}"],
        }
