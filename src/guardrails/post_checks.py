"""
Post-generation rule-based validation.

Runs after all specialist agents complete. Detects inter-agent inconsistencies,
null required fields, verbatim reproduction of retrieved documents, and missing
uncertainty bounds. No LLM calls — pure Python rule checks.

Exported function:
    run_post_checks(state: dict) -> PostCheckResult
"""

import re
from typing import Literal

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------

class CheckFinding(BaseModel):
    check_name: str
    passed: bool
    severity: Literal["info", "warning", "error"]
    detail: str


class PostCheckResult(BaseModel):
    findings: list[CheckFinding]
    passed_all: bool            # True if no "error" severity findings
    error_count: int
    warning_count: int
    should_flag_for_review: bool   # True if error_count > 0 or warning_count >= 3


# ---------------------------------------------------------------------------
# Individual check implementations
# ---------------------------------------------------------------------------

def _check_null_fields(state: dict) -> CheckFinding:
    """
    Check 1: Required agent outputs must not be None when the moonshot gate passed.
    A None output after a successful gate pass indicates a pipeline failure.
    """
    check_name = "null_field_detection"
    try:
        moonshot = state.get("moonshot_evaluation")
        gate_passed = moonshot is not None and getattr(moonshot, "passes_moonshot_gate", False)

        if not gate_passed:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="Moonshot gate did not pass — null field check skipped.",
            )

        required_fields = [
            "technical_output",
            "market_output",
            "risk_output",
            "cost_output",
            "rag_output",
        ]
        null_fields = [f for f in required_fields if state.get(f) is None]

        if null_fields:
            return CheckFinding(
                check_name=check_name,
                passed=False,
                severity="error",
                detail=(
                    f"Required field(s) are None after a successful moonshot gate pass: "
                    f"{', '.join(null_fields)}"
                ),
            )

        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail="All required fields populated after gate pass.",
        )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"Check could not run: {exc}",
        )


def _check_trl_consistency(state: dict) -> CheckFinding:
    """
    Check 2: TRL consistency between technical_output and cost_output assumptions.
    Low-TRL ideas should not have cost assumptions that presuppose commercial scale.
    """
    check_name = "trl_consistency"
    try:
        technical = state.get("technical_output")
        cost = state.get("cost_output")

        if technical is None or cost is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="technical_output or cost_output not available — check skipped.",
            )

        trl = getattr(technical, "trl_level", None)
        assumptions: list[str] = getattr(cost, "assumption_list", []) or []

        if trl is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="trl_level not set — check skipped.",
            )

        if trl >= 5:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail=f"TRL {trl} is ≥5 — no conflict with commercial-scale assumptions.",
            )

        # TRL < 5: check whether cost assumptions imply commercial readiness
        commercial_phrases = {"commercial scale", "proven technology", "commercial-scale"}
        flagged: list[str] = []
        for assumption in assumptions:
            lower = assumption.lower()
            if any(phrase in lower for phrase in commercial_phrases):
                flagged.append(assumption)

        if flagged:
            return CheckFinding(
                check_name=check_name,
                passed=False,
                severity="warning",
                detail=(
                    f"Technical TRL is {trl} (<5) but cost assumptions reference commercial "
                    f"readiness: {flagged}"
                ),
            )

        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"TRL {trl} is consistent with cost assumptions.",
        )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"Check could not run: {exc}",
        )


def _extract_dollar_amounts(text: str) -> list[float]:
    """
    Extract numeric dollar values from a string.
    Handles patterns like: $1.5B, $200M, $10 billion, $3 trillion, $500K, $1,200,000.
    Returns values normalized to USD (float).
    """
    results: list[float] = []

    # Pattern: $<number>[,<digits>]*[.<digits>]? followed by optional multiplier
    pattern = re.compile(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|billion|M|B|T|K|k)?",
        re.IGNORECASE,
    )
    multipliers = {
        "trillion": 1e12,
        "T": 1e12,
        "billion": 1e9,
        "B": 1e9,
        "million": 1e6,
        "M": 1e6,
        "K": 1e3,
        "k": 1e3,
    }

    for match in pattern.finditer(text):
        raw_number = match.group(1).replace(",", "")
        suffix = match.group(2) or ""
        try:
            value = float(raw_number) * multipliers.get(suffix, 1.0)
            results.append(value)
        except ValueError:
            continue

    return results


def _check_market_scale_consistency(state: dict) -> CheckFinding:
    """
    Check 3: Rough sanity check — if risk descriptions mention a market figure
    that differs from market_output.tam_usd by more than 10×, flag it.
    """
    check_name = "market_scale_consistency"
    try:
        market = state.get("market_output")
        risk = state.get("risk_output")

        if market is None or risk is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="market_output or risk_output not available — check skipped.",
            )

        tam: float = getattr(market, "tam_usd", None)
        if tam is None or tam <= 0:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="tam_usd not set or zero — check skipped.",
            )

        technical_risks: list = getattr(risk, "technical_risks", []) or []
        all_risk_text = " ".join(
            getattr(r, "description", "") for r in technical_risks
        )

        extracted = _extract_dollar_amounts(all_risk_text)
        flagged: list[float] = []
        for figure in extracted:
            ratio = max(figure, tam) / max(min(figure, tam), 1.0)
            if ratio > 10:
                flagged.append(figure)

        if flagged:
            flagged_fmt = [f"${v:,.0f}" for v in flagged]
            return CheckFinding(
                check_name=check_name,
                passed=False,
                severity="warning",
                detail=(
                    f"Risk descriptions mention market figure(s) {flagged_fmt} that differ "
                    f"from TAM (${tam:,.0f}) by >10×. Verify market framing is consistent."
                ),
            )

        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"No market-scale inconsistencies detected (TAM=${tam:,.0f}).",
        )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"Check could not run: {exc}",
        )


def _token_overlap_ratio(span_tokens: list[str], chunk_tokens: set[str]) -> float:
    """
    Fraction of span_tokens that appear in chunk_tokens.
    Returns 0.0 if span_tokens is empty.
    """
    if not span_tokens:
        return 0.0
    matches = sum(1 for t in span_tokens if t in chunk_tokens)
    return matches / len(span_tokens)


def _check_verbatim_reproduction(state: dict) -> CheckFinding:
    """
    Check 4: Detect near-verbatim copying from RAG chunks into agent text outputs.
    Uses a sliding 40-token window and flags any window where >80% of tokens
    appear in a single retrieved chunk.
    """
    check_name = "verbatim_reproduction"
    try:
        rag_output = state.get("rag_output")
        if rag_output is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="rag_output not available — check skipped.",
            )

        retrieved_chunks = getattr(rag_output, "retrieved_chunks", []) or []
        if not retrieved_chunks:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="No retrieved chunks — check skipped.",
            )

        # Pre-tokenize all chunks into sets for fast lookup
        chunk_token_sets: list[set[str]] = [
            set(getattr(chunk, "text", "").lower().split())
            for chunk in retrieved_chunks
        ]

        # Collect all long text fields from agent outputs
        agent_text_fields: list[tuple[str, str]] = []
        field_map = {
            "technical_output": ["trl_justification"],
            "market_output": ["competitive_moat"],
            "risk_output": [],  # Risk descriptions are on nested Risk objects
            "cost_output": ["unit_description", "production_scale"],
            "rag_output": [],
        }

        for field_name, attr_names in field_map.items():
            output = state.get(field_name)
            if output is None:
                continue
            for attr in attr_names:
                text = getattr(output, attr, None)
                if text and isinstance(text, str):
                    agent_text_fields.append((f"{field_name}.{attr}", text))

        # Also check risk descriptions from nested Risk objects
        risk_output = state.get("risk_output")
        if risk_output is not None:
            for category in ["technical_risks", "regulatory_risks", "financial_risks", "market_risks"]:
                risks: list = getattr(risk_output, category, []) or []
                for i, risk in enumerate(risks):
                    desc = getattr(risk, "description", None)
                    if desc and isinstance(desc, str):
                        agent_text_fields.append((f"risk_output.{category}[{i}].description", desc))

        window_size = 40
        flagged_fields: list[str] = []

        for field_label, text in agent_text_fields:
            tokens = text.lower().split()
            if len(tokens) < window_size:
                continue

            found_verbatim = False
            for start in range(len(tokens) - window_size + 1):
                window = tokens[start : start + window_size]
                for chunk_tokens in chunk_token_sets:
                    if _token_overlap_ratio(window, chunk_tokens) > 0.80:
                        found_verbatim = True
                        break
                if found_verbatim:
                    break

            if found_verbatim:
                flagged_fields.append(field_label)

        if flagged_fields:
            return CheckFinding(
                check_name=check_name,
                passed=False,
                severity="warning",
                detail=(
                    f"Near-verbatim reproduction of retrieved content detected in: "
                    f"{flagged_fields}. Consider paraphrasing or citing explicitly."
                ),
            )

        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail="No near-verbatim reproduction detected in agent outputs.",
        )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"Check could not run: {exc}",
        )


def _check_uncertainty_bounds(state: dict) -> list[CheckFinding]:
    """
    Check 5: Every agent output must have confidence_level set and assumption_list non-empty.
    Returns one finding per agent output field.
    """
    check_name = "missing_uncertainty_bounds"
    findings: list[CheckFinding] = []

    agent_fields = [
        "technical_output",
        "market_output",
        "risk_output",
        "cost_output",
        "rag_output",
    ]

    for field in agent_fields:
        try:
            output = state.get(field)
            if output is None:
                continue

            confidence = getattr(output, "confidence_level", None)
            assumptions: list = getattr(output, "assumption_list", None)

            if confidence is None:
                findings.append(
                    CheckFinding(
                        check_name=check_name,
                        passed=False,
                        severity="error",
                        detail=f"{field}.confidence_level is None — uncertainty bound missing.",
                    )
                )
            elif assumptions is not None and len(assumptions) == 0:
                findings.append(
                    CheckFinding(
                        check_name=check_name,
                        passed=False,
                        severity="warning",
                        detail=f"{field}.assumption_list is empty — no explicit assumptions declared.",
                    )
                )
            else:
                findings.append(
                    CheckFinding(
                        check_name=check_name,
                        passed=True,
                        severity="info",
                        detail=f"{field}: confidence_level={confidence!r}, {len(assumptions or [])} assumption(s) declared.",
                    )
                )

        except Exception as exc:
            logger.warning(
                "post_check_exception",
                check=check_name,
                field=field,
                error=str(exc),
            )
            findings.append(
                CheckFinding(
                    check_name=check_name,
                    passed=True,
                    severity="info",
                    detail=f"Check could not run for {field}: {exc}",
                )
            )

    if not findings:
        findings.append(
            CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="No agent outputs available to check uncertainty bounds.",
            )
        )

    return findings


def _check_kill_shot_cost_sanity(state: dict) -> CheckFinding:
    """
    Check 6: Kill shot experiment cost should not exceed 1% of total CAPEX.
    An experiment costing more than 1% of CAPEX is not the cheapest falsification test.
    """
    check_name = "kill_shot_cost_sanity"
    try:
        kill_shot = state.get("kill_shot")
        cost_output = state.get("cost_output")

        if kill_shot is None or cost_output is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="kill_shot or cost_output not available — check skipped.",
            )

        experiment_cost: float | None = getattr(kill_shot, "estimated_cost_usd", None)
        capex: float | None = getattr(cost_output, "capex_total_usd", None)

        if experiment_cost is None or capex is None:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="estimated_cost_usd or capex_total_usd not set — check skipped.",
            )

        if capex <= 0:
            return CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail="capex_total_usd is zero or negative — ratio undefined, check skipped.",
            )

        threshold = 0.01 * capex
        if experiment_cost > threshold:
            pct = (experiment_cost / capex) * 100
            return CheckFinding(
                check_name=check_name,
                passed=False,
                severity="warning",
                detail=(
                    f"Kill shot experiment cost (${experiment_cost:,.0f}) is {pct:.1f}% of total "
                    f"CAPEX (${capex:,.0f}), exceeding the 1% ceiling. "
                    f"The experiment may not be the cheapest falsification test."
                ),
            )

        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=(
                f"Kill shot cost (${experiment_cost:,.0f}) is within 1% of CAPEX "
                f"(${capex:,.0f})."
            ),
        )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        return CheckFinding(
            check_name=check_name,
            passed=True,
            severity="info",
            detail=f"Check could not run: {exc}",
        )


def _check_risk_minimum_count(state: dict) -> list[CheckFinding]:
    """
    Check 7: Risk output must have at least 2 technical risks and no empty risk categories.
    """
    check_name = "risk_minimum_count"
    findings: list[CheckFinding] = []

    try:
        risk_output = state.get("risk_output")
        if risk_output is None:
            return [
                CheckFinding(
                    check_name=check_name,
                    passed=True,
                    severity="info",
                    detail="risk_output not available — check skipped.",
                )
            ]

        technical_risks: list = getattr(risk_output, "technical_risks", []) or []
        if len(technical_risks) < 2:
            findings.append(
                CheckFinding(
                    check_name=check_name,
                    passed=False,
                    severity="warning",
                    detail=(
                        f"technical_risks has {len(technical_risks)} item(s) — "
                        f"minimum 2 required for adequate adversarial coverage."
                    ),
                )
            )

        risk_categories = {
            "regulatory_risks": getattr(risk_output, "regulatory_risks", None),
            "financial_risks": getattr(risk_output, "financial_risks", None),
            "market_risks": getattr(risk_output, "market_risks", None),
        }
        for category_name, category_list in risk_categories.items():
            if category_list is None or len(category_list) == 0:
                findings.append(
                    CheckFinding(
                        check_name=check_name,
                        passed=False,
                        severity="warning",
                        detail=f"{category_name} is empty — at least 1 risk required per category.",
                    )
                )

        if not findings:
            findings.append(
                CheckFinding(
                    check_name=check_name,
                    passed=True,
                    severity="info",
                    detail=(
                        f"Risk coverage adequate: {len(technical_risks)} technical risk(s), "
                        f"all categories populated."
                    ),
                )
            )

    except Exception as exc:
        logger.warning("post_check_exception", check=check_name, error=str(exc))
        findings.append(
            CheckFinding(
                check_name=check_name,
                passed=True,
                severity="info",
                detail=f"Check could not run: {exc}",
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_post_checks(state: dict) -> PostCheckResult:
    """
    Execute all post-generation rule checks and aggregate findings.

    Does not call any LLM. Safe to call on a partial state (agents that did not
    run will produce "info" skipped findings rather than errors).
    """
    log = logger.bind(component="post_checks")
    log.info("post_checks_started")

    try:
        findings: list[CheckFinding] = []

        # Check 1: Null fields
        findings.append(_check_null_fields(state))

        # Check 2: TRL consistency
        findings.append(_check_trl_consistency(state))

        # Check 3: Market scale consistency
        findings.append(_check_market_scale_consistency(state))

        # Check 4: Verbatim reproduction
        findings.append(_check_verbatim_reproduction(state))

        # Check 5: Uncertainty bounds (returns a list — one per agent field)
        findings.extend(_check_uncertainty_bounds(state))

        # Check 6: Kill shot cost sanity
        findings.append(_check_kill_shot_cost_sanity(state))

        # Check 7: Risk minimum count (returns a list)
        findings.extend(_check_risk_minimum_count(state))

        error_count = sum(1 for f in findings if f.severity == "error")
        warning_count = sum(1 for f in findings if f.severity == "warning")
        passed_all = error_count == 0
        should_flag = error_count > 0 or warning_count >= 3

        log.info(
            "post_checks_complete",
            total=len(findings),
            errors=error_count,
            warnings=warning_count,
            passed_all=passed_all,
            should_flag=should_flag,
        )

        return PostCheckResult(
            findings=findings,
            passed_all=passed_all,
            error_count=error_count,
            warning_count=warning_count,
            should_flag_for_review=should_flag,
        )

    except Exception as exc:  # noqa: BLE001
        log.exception("post_checks_fatal_error", error=str(exc))
        return PostCheckResult(
            findings=[
                CheckFinding(
                    check_name="post_checks_runner",
                    passed=False,
                    severity="error",
                    detail=f"run_post_checks failed entirely: {exc}",
                )
            ],
            passed_all=False,
            error_count=1,
            warning_count=0,
            should_flag_for_review=True,
        )
