"""
Risk assessment agent.

No external tools — synthesizes adversarial risk analysis directly from the
idea text (and optionally from technical/market outputs already in state).
Uses with_structured_output to produce a RiskAssessment in a single LLM call.
Runs in parallel with technical_agent and market_agent — reads state["idea"],
writes state["risk_output"].

Design note: skipping the tool-calling loop is intentional. Risk identification
is a reasoning task, not a retrieval task. External search results add noise
without improving adversarial coverage for this output type.
"""

import os
from typing import Literal

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class Risk(BaseModel):
    risk_name: str = Field(..., description="Short name for this risk (≤10 words).")
    description: str = Field(
        ...,
        description="Concrete description of how this risk manifests and why it matters.",
    )
    likelihood: Literal["low", "medium", "high"] = Field(
        ...,
        description="Probability this risk materializes before the idea reaches scale.",
    )
    impact: Literal["low", "medium", "high"] = Field(
        ...,
        description="Severity of the consequence if the risk materializes.",
    )
    mitigation: str = Field(
        ...,
        description="Specific, actionable mitigation strategy — not a generic platitude.",
    )


class RiskAssessment(BaseModel):
    technical_risks: list[Risk] = Field(
        ...,
        min_length=2,
        description="Technical risks: at least 2 required.",
    )
    regulatory_risks: list[Risk] = Field(
        ...,
        min_length=1,
        description="Regulatory / compliance risks: at least 1 required.",
    )
    financial_risks: list[Risk] = Field(
        ...,
        min_length=1,
        description="Financial / funding risks: at least 1 required.",
    )
    market_risks: list[Risk] = Field(
        ...,
        min_length=1,
        description="Market / competitive risks: at least 1 required.",
    )
    top_risk: str = Field(
        ...,
        description=(
            "Plain-English description of the single most critical risk — "
            "the one that, if it materializes, kills the venture."
        ),
    )
    overall_risk_level: Literal["low", "medium", "high", "very_high"] = Field(
        ...,
        description="Holistic risk rating across all categories.",
    )
    assumption_list: list[str] = Field(
        ...,
        description="Explicit assumptions baked into this risk assessment.",
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Agent's confidence in its own assessment.",
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description=(
            "References to historical analogues, regulatory documents, or public "
            "case studies used to ground the assessment."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an adversarial risk analyst hired to stress-test a moonshot idea before \
it receives funding. Your job is to find every credible way this idea could fail.

Rules you must follow:
1. Assume Murphy's Law: if something can go wrong, it will.
2. You MUST identify at least 2 technical risks, 1 regulatory risk, 1 financial \
   risk, and 1 market risk. Do not omit any category.
3. A uniformly positive or hedged risk assessment is a failure. If you cannot \
   find real risks, you are not trying hard enough.
4. Each risk must name a specific mechanism, not a vague concern \
   (e.g. "lithium dendrite formation at high C-rates" beats "battery safety").
5. Mitigations must be actionable and specific — not "hire good engineers" or \
   "monitor the market."
6. overall_risk_level should reflect the worst-case plausible scenario across \
   all categories, not the average.
7. Do not use external tools. Reason purely from the idea text and any context \
   provided about technical and market conditions.

Historical pattern to keep in mind: over 90% of deep-tech ventures fail, most \
often due to cost-of-manufacturing surprises, regulatory timelines longer than \
forecast, and the gap between lab performance and production performance. \
Weight these failure modes accordingly.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    idea: str = state["idea"]

    # Incorporate upstream context if available — parallel execution means these
    # may be None when risk_agent runs, which is fine; we handle both cases.
    technical_ctx = state.get("technical_output")
    market_ctx = state.get("market_output")

    log = logger.bind(
        agent="risk_agent",
        idea_preview=idea[:80],
        has_technical_ctx=technical_ctx is not None,
        has_market_ctx=market_ctx is not None,
    )
    log.info("starting risk assessment")

    try:
        model_name = os.getenv("AGENT_MODEL", "gpt-4o-mini")

        structured_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).with_structured_output(RiskAssessment)

        # Build the user message, optionally enriched with upstream context
        user_parts: list[str] = [f"Idea to evaluate:\n{idea}"]

        if technical_ctx is not None:
            user_parts.append(
                f"\nTechnical context (from technical_agent):\n"
                f"  TRL level: {technical_ctx.trl_level}\n"
                f"  Key blockers: {', '.join(technical_ctx.key_blockers)}\n"
                f"  Required breakthroughs: {', '.join(technical_ctx.required_breakthroughs)}"
            )

        if market_ctx is not None:
            user_parts.append(
                f"\nMarket context (from market_agent):\n"
                f"  TAM: ${market_ctx.tam_usd:,.0f}\n"
                f"  Top competitors: {', '.join(market_ctx.top_competitors)}\n"
                f"  Time to market: {market_ctx.time_to_market_years} years"
            )

        user_parts.append(
            "\nNow produce a complete, adversarial RiskAssessment. "
            "Do not soften risks. Every risk must have a specific mechanism and "
            "a concrete mitigation."
        )

        assessment: RiskAssessment = structured_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content="\n".join(user_parts)),
            ]
        )

        log.info(
            "risk assessment complete",
            overall_risk=assessment.overall_risk_level,
            n_technical=len(assessment.technical_risks),
            n_regulatory=len(assessment.regulatory_risks),
            confidence=assessment.confidence_level,
        )
        return {"risk_output": assessment}

    except Exception as exc:  # noqa: BLE001
        log.exception("risk_agent failed", error=str(exc))
        return {
            "risk_output": None,
            "errors": [f"risk_agent: {exc}"],
        }
