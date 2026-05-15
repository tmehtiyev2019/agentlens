"""
LLM-as-Judge scorer for the AgentLens pipeline.

Invoked after all parallel specialist agents and the kill_shot agent have
completed. Scores the full pipeline output across seven criteria using a
DIFFERENT, MORE CAPABLE model than the agents (judge = claude-opus-4-7,
agents = claude-sonnet-4-6) to avoid self-evaluation bias.

Exported function:
    score_output(state: dict) -> JudgeScores
"""

import os
import random
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(..., ge=1.0, le=10.0)
    justification: str  # one sentence grounded in the actual agent output


class JudgeScores(BaseModel):
    moonshot_validity: CriterionScore
    technical_rigor: CriterionScore
    market_realism: CriterionScore
    cost_rigor: CriterionScore
    kill_shot_quality: CriterionScore
    coherence: CriterionScore
    overconfidence_penalty: CriterionScore

    overall_score: float = Field(..., ge=0.0, le=10.0)
    passed_quality_bar: bool
    summary: str  # 2-3 sentence synthesis of the full output


# ---------------------------------------------------------------------------
# Scoring rubric
# ---------------------------------------------------------------------------

# Weights for overall_score (must sum to 1.0 across the positive criteria).
# overconfidence_penalty is a deduction, not a weight.
_WEIGHTS: dict[str, float] = {
    "moonshot_validity": 0.20,
    "technical_rigor":   0.15,
    "market_realism":    0.15,
    "cost_rigor":        0.15,
    "kill_shot_quality": 0.20,
    "coherence":         0.10,
    # remaining 0.05 is offset by overconfidence deduction floor
}

_QUALITY_BAR: float = 6.0


def _compute_overall(scores: JudgeScores) -> float:
    weighted = (
        scores.moonshot_validity.score  * _WEIGHTS["moonshot_validity"]
        + scores.technical_rigor.score  * _WEIGHTS["technical_rigor"]
        + scores.market_realism.score   * _WEIGHTS["market_realism"]
        + scores.cost_rigor.score       * _WEIGHTS["cost_rigor"]
        + scores.kill_shot_quality.score * _WEIGHTS["kill_shot_quality"]
        + scores.coherence.score        * _WEIGHTS["coherence"]
    )
    # Overconfidence penalty: (10 - score) * 0.05 subtracted from overall.
    # A score of 10 means no overconfidence → zero deduction.
    # A score of 1 means severe overconfidence → 0.45 deduction.
    penalty = (10.0 - scores.overconfidence_penalty.score) * 0.05
    return round(max(0.0, min(10.0, weighted - penalty)), 3)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert moonshot evaluator acting as an independent judge. You have \
been given the complete output of a multi-agent analysis pipeline that examined \
an early-stage idea. Your task is to score the OUTPUT — not the idea itself — \
across seven criteria using the rubric below.

CRITICAL RULES:
- Score each criterion INDEPENDENTLY on a scale of 1–10. Do NOT give holistic scores.
- Penalize overconfident claims harshly: any claim without stated uncertainty bounds, \
  any TAM/TRL/cost figure without a stated assumption, or any definitive prediction \
  about markets or technology earns an automatic deduction in overconfidence_penalty.
- Your justification for each criterion must cite a specific element of the agent output \
  (e.g., "the technical agent claims TRL 6 but cites only a blog post").

CRITERION RUBRIC:

1. moonshot_validity (weight 20%)
   - 1: The problem is niche, the solution is incremental, or both.
   - 5: Clear real-world problem, proposed solution offers 2–5x improvement.
   - 10: Genuinely civilizational problem, solution is radically better (10x+), \
     grounded in evidence.

2. technical_rigor (weight 15%)
   - 1: TRL assignment is unsupported or contradicted by cited evidence.
   - 5: TRL is defensible but assumptions are partially missing or hand-wavy.
   - 10: TRL is precisely justified, all key blockers named, required breakthroughs \
     are specific and falsifiable.

3. market_realism (weight 15%)
   - 1: TAM/SAM/SOM figures are unreferenced or SOM exceeds 20% of SAM.
   - 5: TAM and SAM are sourced, SOM is below 10% of SAM with some justification.
   - 10: Bottoms-up market sizing with explicit assumptions, SOM < 5% of SAM, \
     top competitors identified with specific weaknesses.

4. cost_rigor (weight 15%)
   - 1: No explicit cost model, or unit economics are stated without assumptions.
   - 5: CAPEX/OPEX estimates present but ranges are missing or narrow relative to TRL.
   - 10: Full CAPEX/OPEX/unit-cost/break-even model, uncertainty ranges calibrated to TRL, \
     at least 5 explicit assumptions.

5. kill_shot_quality (weight 20%)
   - 1: Experiment described vaguely, no measurable success criterion.
   - 5: Hypothesis is clear but success metric is ambiguous or timeline is unrealistic.
   - 10: Single clear hypothesis, concrete measurable success/failure criteria, \
     realistic timeline, explicit kill conditions, minimal budget footprint.

6. coherence (weight 10%)
   - 1: Agent outputs contradict each other (e.g., cost agent assumes a TRL the \
     technical agent rejected).
   - 5: Minor inconsistencies in assumptions across agents, but no fatal contradictions.
   - 10: All agents tell a single consistent story — TRL, market timeline, cost assumptions, \
     and kill-shot experiment are mutually compatible.

7. overconfidence_penalty (DEDUCTION — not a positive weight)
   Formula: overall_score -= (10 - this_score) * 0.05
   - 1: Pervasive overconfidence — multiple definitive claims without bounds or citations.
   - 5: Some missing uncertainty bounds but most claims are hedged.
   - 10: All claims include uncertainty bounds, confidence levels, or explicit \
     "we don't know" statements where appropriate. Zero penalty applied.

OUTPUT FORMAT:
Produce a JudgeScores object. Compute overall_score as the weighted sum of the \
six positive criteria minus the overconfidence deduction. Set passed_quality_bar \
to True if overall_score >= 6.0. Write a 2-3 sentence summary synthesizing the \
key strengths and the most important improvement needed.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _serialize_field(value: Any) -> str:
    """Convert a Pydantic model, list, or primitive to a readable string."""
    if value is None:
        return "NOT PRODUCED (agent failed or was not reached)"
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, list):
        return "\n".join(f"  - {item}" for item in value)
    return str(value)


def _build_judge_prompt(state: dict) -> str:
    """
    Serializes the pipeline state into a structured prompt.

    Agent sections are shuffled before insertion to mitigate positional bias —
    judges (human and LLM) systematically favour the first and last items they read.
    """
    idea = state.get("idea", "NOT PROVIDED")

    sections = [
        ("MOONSHOT GATE", state.get("moonshot_evaluation")),
        ("TECHNICAL ASSESSMENT", state.get("technical_output")),
        ("MARKET ASSESSMENT", state.get("market_output")),
        ("RISK ASSESSMENT", state.get("risk_output")),
        ("COST ESTIMATION", state.get("cost_output")),
        ("RAG KNOWLEDGE CONTEXT", state.get("rag_output")),
        ("KILL-SHOT EXPERIMENT", state.get("kill_shot")),
    ]

    # Shuffle the specialist agent sections (indices 1–5) to reduce positional bias.
    specialist_sections = sections[1:6]
    random.shuffle(specialist_sections)
    ordered_sections = [sections[0]] + specialist_sections + [sections[6]]

    body_parts = []
    for title, value in ordered_sections:
        body_parts.append(f"--- {title} ---\n{_serialize_field(value)}")

    errors = state.get("errors", [])
    error_block = ""
    if errors:
        error_block = "\n--- PIPELINE ERRORS ---\n" + "\n".join(f"  * {e}" for e in errors)

    return (
        f"IDEA UNDER EVALUATION:\n{idea}\n\n"
        + "\n\n".join(body_parts)
        + error_block
        + "\n\nScore each criterion independently using the rubric in the system prompt."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_output(state: dict) -> JudgeScores:
    """
    Run the LLM-as-judge over the full pipeline state.

    Uses JUDGE_MODEL (default claude-opus-4-7) — always a different, more
    capable model than the specialist agents (AGENT_MODEL = claude-sonnet-4-6).

    The judge uses temperature=0 for determinism. Structured output is enforced
    via with_structured_output so no regex parsing of free-form text is required.

    The overall_score and passed_quality_bar fields returned by the LLM are
    OVERWRITTEN by locally computed values to guarantee formula consistency —
    the LLM cannot be trusted to apply the deduction formula exactly.
    """
    log = logger.bind(
        component="judge",
        idea_preview=str(state.get("idea", ""))[:80],
    )
    log.info("judge scoring started")

    judge_model = os.getenv("JUDGE_MODEL", "claude-opus-4-7")
    agent_model = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
    if judge_model == agent_model:
        log.warning(
            "judge_model equals agent_model — self-evaluation bias risk",
            judge_model=judge_model,
        )

    prompt = _build_judge_prompt(state)

    judge_llm = ChatAnthropic(
        model=judge_model,
        temperature=0,
    ).with_structured_output(JudgeScores)

    try:
        raw_scores: JudgeScores = judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        log.exception("judge_llm_call_failed", error=str(exc))
        raise

    # Recompute overall_score locally — do not trust the LLM's arithmetic.
    overall = _compute_overall(raw_scores)
    passed = overall >= _QUALITY_BAR

    # Pydantic v2: build a new instance with corrected computed fields.
    scores = JudgeScores(
        moonshot_validity=raw_scores.moonshot_validity,
        technical_rigor=raw_scores.technical_rigor,
        market_realism=raw_scores.market_realism,
        cost_rigor=raw_scores.cost_rigor,
        kill_shot_quality=raw_scores.kill_shot_quality,
        coherence=raw_scores.coherence,
        overconfidence_penalty=raw_scores.overconfidence_penalty,
        overall_score=overall,
        passed_quality_bar=passed,
        summary=raw_scores.summary,
    )

    log.info(
        "judge scoring complete",
        overall_score=scores.overall_score,
        passed_quality_bar=scores.passed_quality_bar,
        moonshot_validity=scores.moonshot_validity.score,
        technical_rigor=scores.technical_rigor.score,
        market_realism=scores.market_realism.score,
        cost_rigor=scores.cost_rigor.score,
        kill_shot_quality=scores.kill_shot_quality.score,
        coherence=scores.coherence.score,
        overconfidence_penalty=scores.overconfidence_penalty.score,
    )

    return scores
