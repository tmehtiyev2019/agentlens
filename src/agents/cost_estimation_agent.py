"""
Cost Estimation Agent — parallel specialist in the AgentLens pipeline.

Three-step pattern:
  Step 1: Literature search via ArxivQueryRun — collect CAPEX/OPEX benchmarks.
  Step 2: Numeric computation via PythonREPLTool — break-even, unit cost, NPV.
  Step 3: Structured synthesis via with_structured_output(CostEstimation).

ALL arithmetic is delegated to PythonREPLTool. The LLM never does arithmetic.
"""

import os
import re
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class CostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    category: str
    amount_usd: float
    basis: str  # e.g. "$/unit from ArXiv 2023", "industry benchmark"


class CostEstimation(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Capital expenditure
    capex_total_usd: float
    capex_breakdown: list[CostBreakdown]

    # Operating expenditure (annual)
    opex_annual_usd: float
    opex_breakdown: list[CostBreakdown]

    # Unit economics
    unit_cost_usd: float
    unit_description: str       # e.g. "per kg H2 produced"
    production_scale: str       # assumed scale, e.g. "1000 units/year"

    # Break-even
    break_even_units: float
    break_even_years: float

    # Ranges (reflect uncertainty)
    capex_low_usd: float
    capex_high_usd: float
    opex_low_usd: float
    opex_high_usd: float

    assumption_list: list[str] = Field(min_length=5)
    confidence_level: Literal["low", "medium", "high"]
    evidence_citations: list[str]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GATHER_PROMPT = """\
You are a techno-economic analyst. Your task is to search published academic literature \
for CAPEX and OPEX cost benchmarks relevant to the product idea below.

For each search:
- Look for reported capital expenditure (equipment, facilities, installation) and \
  operating expenditure (labor, energy, maintenance, feedstocks) figures.
- Prefer papers that give dollar values per unit of output (e.g., $/kg, $/MWh, $/unit).
- Note the year, technology readiness level (TRL), and production scale of each figure.
- If multiple estimates exist, capture the range (low / central / high).

Conduct up to 4 ArXiv searches. After the tool calls, write a concise evidence summary \
that lists ALL numeric cost figures found, with their sources and assumptions.\
"""

CALCULATION_PROMPT = """\
You are a techno-economic analyst. You have gathered cost benchmarks from the literature. \
Your task is to write Python code that computes the following numeric outputs from those benchmarks:

1. capex_total_usd           — total capital expenditure (single best estimate)
2. opex_annual_usd           — annual operating expenditure (single best estimate)
3. unit_cost_usd             — cost per unit of output (amortize capex over 10 years, add opex)
4. break_even_units          — units required to recover capex at the assumed selling price
5. break_even_years          — years required to recover capex at expected annual production
6. capex_low_usd             — capex low bound (−40% for TRL < 5, −20% for TRL ≥ 5)
7. capex_high_usd            — capex high bound (+40% for TRL < 5, +20% for TRL ≥ 5)
8. opex_low_usd              — opex low bound (same uncertainty fractions as capex)
9. opex_high_usd             — opex high bound

Rules:
- Use only the numeric values from the evidence summary. Set explicit Python variables \
  with inline comments showing the source.
- Assume a 10-year capex amortization period unless a paper specifies otherwise.
- Assume selling_price = unit_cost_usd * 2 for break-even unless the idea implies a \
  specific price point.
- Print every result with a clear label so it can be parsed.
- Do NOT import anything that is not in the Python standard library — the REPL is sandboxed.
- Write one self-contained code block.

Here is the evidence summary:
{evidence_summary}

Now write the Python code.\
"""

SYNTHESIZE_PROMPT = """\
You are a techno-economic analyst. You have gathered literature benchmarks and computed \
numeric results. Your task is to produce a structured CostEstimation report.

Use ONLY the numeric values from the computed results below — do not re-derive or \
recalculate anything. Populate all fields exactly as computed.

RULES:
- capex_breakdown and opex_breakdown must reflect the cost categories found in the literature.
- unit_description must be specific (e.g. "per kg H2 produced", not "per unit").
- production_scale must state the assumed annual production volume.
- assumption_list must contain AT LEAST 5 explicit assumptions (production scale, location, \
  discount rate, amortization period, selling price basis, learning curve, etc.).
- confidence_level: "low" if TRL < 5, "medium" if TRL 5–7, "high" if TRL ≥ 8.
- evidence_citations must list every paper title or URL that provided numeric data.
- Do not speculate beyond what the evidence and computation support.\
"""


# ---------------------------------------------------------------------------
# Agent run function
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    idea: str = state["idea"]
    log = logger.bind(agent="cost_estimation_agent", idea_snippet=idea[:80])

    try:
        # Step 1: Literature search
        evidence_summary = _gather_cost_benchmarks(idea, log)

        # Step 2: Numeric calculation via PythonREPLTool
        computed_results = _run_numeric_calculations(evidence_summary, log)

        # Step 3: Structured synthesis
        estimation = _synthesize(idea, evidence_summary, computed_results, log)

        log.info(
            "cost_estimation_complete",
            capex=estimation.capex_total_usd,
            opex_annual=estimation.opex_annual_usd,
            unit_cost=estimation.unit_cost_usd,
            break_even_years=estimation.break_even_years,
            confidence=estimation.confidence_level,
        )
        return {"cost_output": estimation}

    except Exception as exc:
        log.error("cost_estimation_failed", error=str(exc))
        return {
            "cost_output": None,
            "errors": [f"cost_estimation_agent: {exc}"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gather_cost_benchmarks(idea: str, log) -> str:
    """
    Step 1: Run an agentic tool-use loop with ArxivQueryRun to collect
    CAPEX/OPEX cost benchmarks from published literature. Max 4 tool calls.
    """
    arxiv = ArxivQueryRun()
    tools = [arxiv]
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

    for call_index in range(4):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            log.debug("cost_evidence_loop_stopped", calls_made=call_index)
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            try:
                result = tool_map[tool_name].invoke(tc["args"])
                content = str(result)[:2000]
            except Exception as exc:
                content = f"[tool_error: {exc}]"
            messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    # Return the last non-empty text content from the LLM
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content

    return "No cost benchmark evidence produced."


def _run_numeric_calculations(evidence_summary: str, log) -> str:
    """
    Step 2: Ask the LLM to write Python code, then execute it with PythonREPLTool.
    Returns the stdout of the executed code (the labeled numeric results).

    The LLM is only used to translate evidence into code — the arithmetic is
    performed entirely by the Python interpreter, not the LLM.
    """
    python_repl = PythonREPLTool()

    # Ask the LLM to write calculation code grounded in the evidence summary
    code_request_llm = ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    code_response = code_request_llm.invoke(
        [
            SystemMessage(content=CALCULATION_PROMPT.format(evidence_summary=evidence_summary)),
            HumanMessage(content="Write the Python calculation code now."),
        ]
    )

    # Extract the code block from the LLM response
    raw_response = code_response.content if isinstance(code_response.content, str) else ""
    code = _extract_code_block(raw_response)

    if not code:
        # Fallback: treat the entire response as code
        code = raw_response.strip()

    log.debug("cost_repl_code_ready", code_length=len(code))

    # Execute the code in the sandboxed REPL — all arithmetic happens here
    repl_output = python_repl.run(code)
    log.debug("cost_repl_output", output_snippet=str(repl_output)[:200])

    return str(repl_output)


def _extract_code_block(text: str) -> str:
    """
    Pull the first ```python ... ``` or ``` ... ``` block from LLM text.
    Returns an empty string if no fenced block is found.
    """
    # Try ```python first, then generic ```
    for pattern in (r"```python\s*(.*?)```", r"```\s*(.*?)```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _synthesize(
    idea: str,
    evidence_summary: str,
    computed_results: str,
    log,
) -> CostEstimation:
    """
    Step 3: Combine evidence and computed results into a typed CostEstimation
    using with_structured_output so schema conformance is enforced by the SDK.
    """
    llm = ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    llm_structured = llm.with_structured_output(CostEstimation)

    synthesis_input = (
        f"IDEA:\n{idea}\n\n"
        f"LITERATURE EVIDENCE:\n{evidence_summary}\n\n"
        f"COMPUTED NUMERIC RESULTS:\n{computed_results}"
    )

    log.debug("cost_synthesis_start")
    estimation: CostEstimation = llm_structured.invoke(
        [
            SystemMessage(content=SYNTHESIZE_PROMPT),
            HumanMessage(content=synthesis_input),
        ]
    )
    return estimation
