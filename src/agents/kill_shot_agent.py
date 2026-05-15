"""
Kill Shot agent — final sequential stage in the AgentLens pipeline.

Runs after all 5 parallel specialist agents AND after the human review HITL
pause. Reads the full accumulated state, identifies the single most critical
assumption that if false kills the idea, and designs the cheapest binary
experiment to test it.

State fields read:
    idea, moonshot_evaluation, technical_output, market_output, risk_output,
    cost_output, rag_output, human_comment

State fields written:
    kill_shot  (KillShotExperiment | None)
    errors     (list[str], appended on exception)

Design notes:
- No tool-calling loop. One structured synthesis call after context assembly.
- PythonREPLTool is used only for the cost sanity check (arithmetic, not
  reasoning) — the LLM never does the arithmetic.
- The HITL human_comment, when present, is incorporated verbatim into the
  context and the system prompt requires explicit acknowledgment.
- Degrades gracefully when all parallel agents failed (confidence_level="low").
"""

import os
import re
from typing import Literal

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class KillShotExperiment(BaseModel):
    critical_assumption: str = Field(
        ...,
        description=(
            "The ONE assumption everything else depends on. If this is false, "
            "the idea cannot proceed regardless of all other factors."
        ),
    )
    why_this_assumption: str = Field(
        ...,
        description=(
            "Why this specific assumption was chosen over all others — what "
            "makes it the load-bearing one, not the others."
        ),
    )
    experiment_description: str = Field(
        ...,
        description=(
            "Plain-language description of what to build or test. Must be the "
            "CHEAPEST possible test of the critical assumption — minimise cost "
            "and time, maximise signal."
        ),
    )
    success_criteria: str = Field(
        ...,
        description=(
            "The specific, measurable outcome that means the assumption holds "
            "and the idea can advance to the next stage."
        ),
    )
    failure_criteria: str = Field(
        ...,
        description=(
            "The specific, measurable outcome that falsifies the assumption and "
            "kills the idea. Must be as concrete as success_criteria."
        ),
    )
    estimated_cost_usd: float = Field(
        ...,
        description="Best-estimate total cost of running this experiment in USD.",
    )
    estimated_duration_weeks: int = Field(
        ...,
        description="Calendar weeks required to reach a decisive result.",
    )
    required_resources: list[str] = Field(
        ...,
        description="Concrete list of people, equipment, and materials needed.",
    )
    informed_by_portfolio: bool = Field(
        ...,
        description=(
            "True if portfolio RAG chunks materially informed the choice of "
            "critical assumption or experiment design."
        ),
    )
    portfolio_reference: str | None = Field(
        default=None,
        description=(
            "Name of the past project from the portfolio that is most relevant, "
            "or None if not applicable."
        ),
    )
    assumption_list: list[str] = Field(
        ...,
        description=(
            "Explicit assumptions made when designing this experiment — e.g., "
            "lab access, regulatory environment, baseline technology availability."
        ),
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        ...,
        description=(
            "'high' when multiple agent outputs are available and consistent; "
            "'medium' when some outputs are missing or conflicting; "
            "'low' when operating on idea text alone (all agents failed)."
        ),
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a moonshot kill-shot analyst. Your role is the final adversarial check \
before a project advances to funding: identify the SINGLE most critical \
assumption the idea rests on, and design the CHEAPEST binary experiment that \
could falsify it within weeks, not years.

Rules you must follow:

1. SINGLE ASSUMPTION: Output exactly one critical assumption — the one that, \
   if false, makes every other analysis irrelevant. Do not hedge with a list.

2. CHEAPEST TEST: The experiment must be the minimum viable falsification test. \
   A smoke test, a paper study, a bench experiment, or a customer interview can \
   all qualify — choose the one with the lowest cost and shortest timeline that \
   still produces a decisive signal.

3. BINARY OUTCOME: The experiment must have a clear success criterion and a \
   clear failure criterion. Ambiguous outcomes are not acceptable. The outcome \
   either kills the idea or validates the path forward.

4. ASSUMPTION PROVENANCE: Prefer assumptions drawn from:
   a) technical_output.key_blockers — these are the hardest unsolved problems
   b) risk_output.top_risk — the adversarial worst-case view
   c) cost_output.break_even_years — if > 15 years, economic viability is itself
      the critical assumption
   d) portfolio_chunks with outcome="killed" — learn from what already failed
   e) moonshot_evaluation.problem_evidence — if the problem isn't real, nothing else matters

5. HUMAN COMMENT: If a human comment (domain expert correction) is present in \
   the context, you MUST incorporate it and explicitly acknowledge it in your \
   choice of critical assumption. Human correction overrides agent inference.

6. PORTFOLIO LESSONS: If portfolio chunks contain documents where outcome="killed", \
   extract the assumption that killed them. Weight this heavily — it is empirical \
   evidence, not inference.

7. DEGRADED MODE: If all parallel agents failed and you are operating on idea \
   text alone, still produce a complete output. Set confidence_level="low" and \
   be explicit in assumption_list that you lacked agent outputs.

8. COST DISCIPLINE: estimated_cost_usd must be << 1% of the project's total CAPEX \
   (if known). The kill shot experiment is a cheap gate, not a project in itself.

Output a complete KillShotExperiment. Every field must be populated.\
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(state: dict) -> str:
    """
    Assemble a structured plain-text context block from all available state
    fields. Fields that are None are noted as unavailable so the LLM knows
    what it is missing rather than hallucinating values.
    """
    lines: list[str] = []

    idea: str = state.get("idea", "")
    lines.append(f"IDEA:\n{idea}\n")

    # Human expert comment — surfaced first because it has highest priority
    human_comment: str | None = state.get("human_comment")
    if human_comment:
        lines.append(
            f"HUMAN EXPERT COMMENT (domain correction — must be incorporated):\n"
            f"{human_comment}\n"
        )
    else:
        lines.append("HUMAN EXPERT COMMENT: none provided\n")

    # Moonshot evaluation
    moonshot = state.get("moonshot_evaluation")
    if moonshot is not None:
        lines.append(
            f"MOONSHOT GATE:\n"
            f"  passes: {moonshot.passes_moonshot_gate}\n"
            f"  trl: {moonshot.technology_trl}\n"
            f"  problem_is_real: {moonshot.problem_is_real}\n"
            f"  problem_evidence: {'; '.join(moonshot.problem_evidence[:3])}\n"
            f"  gate_failure_reason: {moonshot.gate_failure_reason or 'none'}\n"
            f"  confidence: {moonshot.confidence_level}\n"
        )
    else:
        lines.append("MOONSHOT GATE: unavailable (agent failed)\n")

    # Technical assessment
    technical = state.get("technical_output")
    if technical is not None:
        lines.append(
            f"TECHNICAL ASSESSMENT:\n"
            f"  trl_level: {technical.trl_level}\n"
            f"  trl_justification: {technical.trl_justification}\n"
            f"  key_blockers: {'; '.join(technical.key_blockers)}\n"
            f"  required_breakthroughs: {'; '.join(technical.required_breakthroughs)}\n"
            f"  time_to_prototype_years: {technical.time_to_prototype_years}\n"
            f"  confidence: {technical.confidence_level}\n"
        )
    else:
        lines.append("TECHNICAL ASSESSMENT: unavailable (agent failed)\n")

    # Market assessment
    market = state.get("market_output")
    if market is not None:
        lines.append(
            f"MARKET ASSESSMENT:\n"
            f"  tam_usd: ${market.tam_usd:,.0f}\n"
            f"  sam_usd: ${market.sam_usd:,.0f}\n"
            f"  som_usd: ${market.som_usd:,.0f}\n"
            f"  top_competitors: {'; '.join(market.top_competitors)}\n"
            f"  time_to_market_years: {market.time_to_market_years}\n"
            f"  confidence: {market.confidence_level}\n"
        )
    else:
        lines.append("MARKET ASSESSMENT: unavailable (agent failed)\n")

    # Risk assessment
    risk = state.get("risk_output")
    if risk is not None:
        tech_risk_names = [r.risk_name for r in risk.technical_risks]
        lines.append(
            f"RISK ASSESSMENT:\n"
            f"  top_risk: {risk.top_risk}\n"
            f"  overall_risk_level: {risk.overall_risk_level}\n"
            f"  technical_risks: {'; '.join(tech_risk_names)}\n"
            f"  confidence: {risk.confidence_level}\n"
        )
    else:
        lines.append("RISK ASSESSMENT: unavailable (agent failed)\n")

    # Cost estimation
    cost = state.get("cost_output")
    if cost is not None:
        lines.append(
            f"COST ESTIMATION:\n"
            f"  capex_total_usd: ${cost.capex_total_usd:,.0f}\n"
            f"  capex_range: ${cost.capex_low_usd:,.0f} – ${cost.capex_high_usd:,.0f}\n"
            f"  opex_annual_usd: ${cost.opex_annual_usd:,.0f}\n"
            f"  unit_cost_usd: ${cost.unit_cost_usd:,.2f} {cost.unit_description}\n"
            f"  break_even_years: {cost.break_even_years}\n"
            f"  confidence: {cost.confidence_level}\n"
        )
    else:
        lines.append("COST ESTIMATION: unavailable (agent failed)\n")

    # RAG context — portfolio chunks are the highest-value signal here
    rag = state.get("rag_output")
    if rag is not None and rag.portfolio_chunks:
        portfolio_lines: list[str] = []
        for chunk in rag.portfolio_chunks[:5]:
            outcome = chunk.metadata.get("outcome", "unknown")
            project = chunk.metadata.get("project_name", chunk.source_file)
            portfolio_lines.append(
                f"  [{project} | outcome={outcome} | score={chunk.similarity_score}]\n"
                f"  {chunk.text[:400]}"
            )
        lines.append(
            f"PORTFOLIO CONTEXT ({len(rag.portfolio_chunks)} chunks):\n"
            + "\n\n".join(portfolio_lines)
            + "\n"
        )
    elif rag is not None:
        lines.append(
            f"PORTFOLIO CONTEXT: no portfolio chunks above similarity threshold "
            f"(grounded={rag.grounded}, total_retrieved={rag.total_retrieved})\n"
        )
    else:
        lines.append("PORTFOLIO CONTEXT: unavailable (agent failed)\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cost sanity check via PythonREPLTool
# ---------------------------------------------------------------------------

def _sanity_check_cost(
    experiment_cost: float,
    capex_total: float | None,
    log,
) -> str:
    """
    Use PythonREPLTool to verify that the kill-shot experiment cost is << 1% of
    total CAPEX. Returns a plain-text sanity report string that is logged but
    does NOT modify the structured output — the LLM already produced the schema.

    The arithmetic is done by the Python interpreter, not the LLM.
    """
    if capex_total is None or capex_total <= 0:
        return "CAPEX unavailable — cost sanity check skipped."

    code = (
        f"experiment_cost = {experiment_cost}\n"
        f"capex_total = {capex_total}\n"
        f"ratio_pct = (experiment_cost / capex_total) * 100\n"
        f"threshold_pct = 1.0\n"
        f"status = 'PASS' if ratio_pct < threshold_pct else 'WARN'\n"
        f"print(f'experiment_cost: ${experiment_cost:,.2f}')\n"
        f"print(f'capex_total:     ${capex_total:,.0f}')\n"
        f"print(f'ratio:           {{ratio_pct:.4f}}%')\n"
        f"print(f'threshold:       {{threshold_pct}}%')\n"
        f"print(f'status:          {{status}}')\n"
    )

    try:
        repl = PythonREPLTool()
        output = repl.run(code)
        log.info("kill_shot_cost_sanity_check", output=str(output).strip())
        return str(output).strip()
    except Exception as exc:
        log.warning("kill_shot_cost_sanity_check_failed", error=str(exc))
        return f"Cost sanity check failed: {exc}"


# ---------------------------------------------------------------------------
# Confidence inference
# ---------------------------------------------------------------------------

def _infer_confidence(state: dict) -> Literal["low", "medium", "high"]:
    """
    Derive the appropriate confidence level from how many agent outputs are
    available. The LLM will ultimately set this in the schema; this helper
    provides a sensible prior that can be included in the context if needed.
    """
    available = sum(
        1
        for key in ("technical_output", "market_output", "risk_output", "cost_output", "rag_output")
        if state.get(key) is not None
    )
    if available == 0:
        return "low"
    if available <= 2:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    idea: str = state.get("idea", "")
    log = logger.bind(agent="kill_shot_agent", idea_preview=idea[:80])
    log.info("starting kill shot synthesis")

    try:
        model_name = os.getenv("AGENT_MODEL", "gpt-4o-mini")

        # Step 1: Assemble full context from all available state fields
        context = _build_context(state)
        prior_confidence = _infer_confidence(state)
        human_comment: str | None = state.get("human_comment")

        log.debug(
            "context assembled",
            context_length=len(context),
            prior_confidence=prior_confidence,
            has_human_comment=human_comment is not None,
        )

        # Build the user message. The human comment gets an explicit nudge so
        # the model never buries it under agent outputs.
        user_parts: list[str] = [context]

        if human_comment:
            user_parts.append(
                "\nREMINDER: A domain expert has provided a correction above "
                "(HUMAN EXPERT COMMENT). You MUST incorporate this into your "
                "choice of critical assumption and explicitly acknowledge it in "
                "your why_this_assumption field."
            )

        user_parts.append(
            f"\nPrior confidence estimate (based on agent availability): "
            f"{prior_confidence}. Adjust your confidence_level field based on "
            f"the actual quality and consistency of the evidence above."
        )

        user_parts.append(
            "\nNow produce a complete KillShotExperiment. "
            "The experiment must be the cheapest possible binary test. "
            "One assumption. One experiment. Decisive outcome."
        )

        # Step 2: Structured synthesis — no tools bound; forces schema output
        structured_llm = ChatOpenAI(
            model=model_name,
            temperature=0,
        ).with_structured_output(KillShotExperiment)

        experiment: KillShotExperiment = structured_llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content="\n".join(user_parts)),
            ]
        )

        # Step 3: Cost sanity check via PythonREPLTool (arithmetic only)
        cost_output = state.get("cost_output")
        capex_total = cost_output.capex_total_usd if cost_output is not None else None
        sanity_report = _sanity_check_cost(
            experiment_cost=experiment.estimated_cost_usd,
            capex_total=capex_total,
            log=log,
        )

        log.info(
            "kill shot complete",
            critical_assumption_preview=experiment.critical_assumption[:120],
            estimated_cost_usd=experiment.estimated_cost_usd,
            estimated_duration_weeks=experiment.estimated_duration_weeks,
            informed_by_portfolio=experiment.informed_by_portfolio,
            confidence_level=experiment.confidence_level,
            cost_sanity=sanity_report,
        )

        return {"kill_shot": experiment}

    except Exception as exc:  # noqa: BLE001
        log.exception("kill_shot_agent failed", error=str(exc))
        return {
            "kill_shot": None,
            "errors": [f"kill_shot_agent: {exc}"],
        }
