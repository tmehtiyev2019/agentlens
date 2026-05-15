"""
Benchmark runner for the AgentLens multi-agent pipeline.

Sweeps a configuration matrix (agent model × RAG on/off) across a fixed set of
test cases, scores each run with the LLM-as-judge and RAGAS, and produces:
  - A comparison table printed to stdout
  - A JSON results file saved to benchmarks/results/run_{timestamp}.json

Usage:
    python -m src.evaluation.benchmark

Design notes:
- Environment variables (AGENT_MODEL, JUDGE_MODEL) must be set BEFORE importing
  the graph because agents read os.getenv() at import time.
- When rag_enabled=False the VectorStore.query method is monkeypatched to return
  [] for the duration of the run, then restored. This isolates the RAG contribution
  to judge scores and RAGAS metrics without modifying agent code.
- A single failed run does not abort the sweep — the error is logged and the result
  is recorded with partial metrics so the table remains complete.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration matrix
# ---------------------------------------------------------------------------

CONFIGS: list[dict[str, Any]] = [
    {
        "name": "sonnet-rag-on",
        "agent_model": "claude-sonnet-4-6",
        "judge_model": "claude-opus-4-7",
        "rag_enabled": True,
    },
    {
        "name": "sonnet-rag-off",
        "agent_model": "claude-sonnet-4-6",
        "judge_model": "claude-opus-4-7",
        "rag_enabled": False,
    },
    {
        "name": "opus-rag-on",
        "agent_model": "claude-opus-4-7",
        "judge_model": "claude-opus-4-7",
        "rag_enabled": True,
    },
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_TEST_CASES_PATH = _PROJECT_ROOT / "benchmarks" / "test_cases.json"
_RESULTS_DIR = _PROJECT_ROOT / "benchmarks" / "results"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    test_case_id: str
    config_name: str
    agent_model: str
    moonshot_gate_passed: bool
    judge_scores: dict               # JudgeScores serialized as dict
    overall_score: float
    passed_quality_bar: bool
    ragas_faithfulness: float
    ragas_context_precision: float
    latency_seconds: float
    error_count: int
    run_id: str                      # uuid4
    timestamp: str                   # ISO 8601


# ---------------------------------------------------------------------------
# RAG disable helpers
# ---------------------------------------------------------------------------


def _patch_rag_disabled(vector_store_module: Any) -> Any:
    """Replace VectorStore.query with a stub returning [] and return the original."""
    original = vector_store_module.VectorStore.query

    def _stub(self, *args, **kwargs) -> list:
        return []

    vector_store_module.VectorStore.query = _stub
    return original


def _restore_rag(vector_store_module: Any, original: Any) -> None:
    """Restore the original VectorStore.query."""
    vector_store_module.VectorStore.query = original


# ---------------------------------------------------------------------------
# Graph module loader
#
# The orchestrator reads os.getenv("AGENT_MODEL") at import time when it
# constructs the ChatAnthropic clients. Because Python caches imported modules,
# we force a full reload of the agent subpackage for each configuration so that
# the model env var is picked up fresh each time.
# ---------------------------------------------------------------------------

_AGENT_MODULES = [
    "src.agents.moonshot_evaluator",
    "src.agents.technical_agent",
    "src.agents.market_agent",
    "src.agents.risk_agent",
    "src.agents.cost_estimation_agent",
    "src.agents.rag_agent",
    "src.agents.kill_shot_agent",
    "src.agents.orchestrator",
]


def _load_graph_for_config(config: dict[str, Any]) -> Any:
    """
    Set environment variables, flush cached agent modules, and return a freshly
    compiled tea_graph for this configuration.

    The reload order matters: leaf agents must be reloaded before the orchestrator
    that imports them.
    """
    os.environ["AGENT_MODEL"] = config["agent_model"]
    os.environ["JUDGE_MODEL"] = config["judge_model"]

    for module_name in _AGENT_MODULES:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                # kill_shot_agent may not exist yet; orchestrator handles this
                pass

    orchestrator_mod = sys.modules.get("src.agents.orchestrator")
    if orchestrator_mod is None:
        orchestrator_mod = importlib.import_module("src.agents.orchestrator")

    return orchestrator_mod.tea_graph


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _build_initial_state(idea: str) -> dict[str, Any]:
    return {
        "idea": idea,
        "errors": [],
        "intent_safe": None,
        "moonshot_evaluation": None,
        "technical_output": None,
        "market_output": None,
        "risk_output": None,
        "cost_output": None,
        "rag_output": None,
        "human_decision": "approved",   # auto-approve for benchmarking
        "human_comment": None,
        "kill_shot": None,
        "grounded": False,
    }


def _run_single(
    test_case: dict[str, Any],
    config: dict[str, Any],
) -> BenchmarkResult:
    """Execute one test-case × config combination and return a BenchmarkResult."""
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    log = logger.bind(
        run_id=run_id,
        config=config["name"],
        test_case=test_case["id"],
    )
    log.info("benchmark_run_start")

    # Lazily import scoring functions here (after env vars are set by the caller)
    from src.evaluation.judge import score_output
    from src.evaluation.metrics import compute_rag_metrics
    import src.knowledge_base.vector_store as vs_module

    # ------------------------------------------------------------------
    # Load graph (with env vars already set by caller)
    # ------------------------------------------------------------------
    try:
        tea_graph = _load_graph_for_config(config)
    except Exception as exc:
        log.error("graph_load_failed", error=str(exc))
        return _error_result(test_case["id"], config, run_id, timestamp, error=str(exc))

    # ------------------------------------------------------------------
    # Optionally disable RAG by monkeypatching VectorStore.query
    # ------------------------------------------------------------------
    original_query = None
    if not config["rag_enabled"]:
        original_query = _patch_rag_disabled(vs_module)
        log.debug("rag_disabled_for_run")

    try:
        initial_state = _build_initial_state(test_case["idea"])
        t0 = time.perf_counter()

        # LangGraph compiled graphs with interrupt_after return after the
        # interrupt point. We must resume to run kill_shot. We stream events
        # and collect the final state across both invocations.
        state_after_interrupt = tea_graph.invoke(initial_state)

        # Resume past the human_review interrupt (human_decision already set)
        final_state = tea_graph.invoke(None, state_after_interrupt.get("configurable", {}))
        if final_state is None:
            # Fallback: use the state from the first invocation (graph may have
            # terminated early via rejection_report before the interrupt)
            final_state = state_after_interrupt

        latency = round(time.perf_counter() - t0, 2)

    except Exception as exc:
        log.error("graph_invoke_failed", error=str(exc))
        return _error_result(test_case["id"], config, run_id, timestamp, error=str(exc))
    finally:
        if original_query is not None:
            _restore_rag(vs_module, original_query)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    try:
        judge_scores = score_output(final_state)
    except Exception as exc:
        log.error("judge_scoring_failed", error=str(exc))
        return _error_result(test_case["id"], config, run_id, timestamp, error=str(exc))

    try:
        rag_metrics = compute_rag_metrics(final_state)
    except Exception as exc:
        log.warning("rag_metrics_failed", error=str(exc))
        from src.evaluation.metrics import _zero_metrics
        rag_metrics = _zero_metrics()

    # ------------------------------------------------------------------
    # Derive moonshot gate result
    # ------------------------------------------------------------------
    moonshot_eval = final_state.get("moonshot_evaluation")
    moonshot_gate_passed = bool(
        moonshot_eval and getattr(moonshot_eval, "passes_moonshot_gate", False)
    )

    errors: list[str] = final_state.get("errors", [])

    result = BenchmarkResult(
        test_case_id=test_case["id"],
        config_name=config["name"],
        agent_model=config["agent_model"],
        moonshot_gate_passed=moonshot_gate_passed,
        judge_scores=judge_scores.model_dump(),
        overall_score=judge_scores.overall_score,
        passed_quality_bar=judge_scores.passed_quality_bar,
        ragas_faithfulness=rag_metrics.faithfulness,
        ragas_context_precision=rag_metrics.context_precision,
        latency_seconds=latency,
        error_count=len(errors),
        run_id=run_id,
        timestamp=timestamp,
    )

    log.info(
        "benchmark_run_complete",
        overall_score=result.overall_score,
        passed_quality_bar=result.passed_quality_bar,
        ragas_faithfulness=result.ragas_faithfulness,
        latency_seconds=result.latency_seconds,
        error_count=result.error_count,
    )

    return result


def _error_result(
    test_case_id: str,
    config: dict[str, Any],
    run_id: str,
    timestamp: str,
    error: str,
) -> BenchmarkResult:
    """Return a zeroed BenchmarkResult representing a failed run."""
    logger.error(
        "benchmark_run_error",
        test_case_id=test_case_id,
        config=config["name"],
        error=error,
    )
    return BenchmarkResult(
        test_case_id=test_case_id,
        config_name=config["name"],
        agent_model=config["agent_model"],
        moonshot_gate_passed=False,
        judge_scores={},
        overall_score=0.0,
        passed_quality_bar=False,
        ragas_faithfulness=0.0,
        ragas_context_precision=0.0,
        latency_seconds=0.0,
        error_count=1,
        run_id=run_id,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "config": 20,
    "test_case": 24,
    "score": 8,
    "quality": 9,
    "faithfulness": 14,
    "latency": 10,
}


def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Config':<{_COL_WIDTHS['config']}} | "
        f"{'Test Case':<{_COL_WIDTHS['test_case']}} | "
        f"{'Score':<{_COL_WIDTHS['score']}} | "
        f"{'Quality':<{_COL_WIDTHS['quality']}} | "
        f"{'Faithfulness':<{_COL_WIDTHS['faithfulness']}} | "
        f"{'Latency'}"
    )
    divider = "-" * len(header)

    print()
    print(header)
    print(divider)

    for r in results:
        quality_str = "PASS" if r.passed_quality_bar else "FAIL"
        error_suffix = f"  [errors: {r.error_count}]" if r.error_count else ""
        row = (
            f"{r.config_name:<{_COL_WIDTHS['config']}} | "
            f"{r.test_case_id:<{_COL_WIDTHS['test_case']}} | "
            f"{r.overall_score:<{_COL_WIDTHS['score']}.1f} | "
            f"{quality_str:<{_COL_WIDTHS['quality']}} | "
            f"{r.ragas_faithfulness:<{_COL_WIDTHS['faithfulness']}.2f} | "
            f"{r.latency_seconds:.1f}s"
            f"{error_suffix}"
        )
        print(row)

    print()
    pass_count = sum(1 for r in results if r.passed_quality_bar)
    print(f"Summary: {pass_count}/{len(results)} runs passed quality bar (overall_score >= 6.0)")
    print()


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(results: list[BenchmarkResult], run_timestamp: str) -> pathlib.Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize timestamp for use in a filename
    safe_ts = run_timestamp.replace(":", "-").replace("+", "").replace(".", "-")[:23]
    output_path = _RESULTS_DIR / f"run_{safe_ts}.json"

    payload = {
        "benchmark_run_timestamp": run_timestamp,
        "configs_swept": [c["name"] for c in CONFIGS],
        "total_runs": len(results),
        "passed_quality_bar": sum(1 for r in results if r.passed_quality_bar),
        "results": [r.model_dump() for r in results],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_benchmark() -> list[BenchmarkResult]:
    """
    Load test cases, sweep all config × test-case combinations, score, and report.

    Returns the full list of BenchmarkResult objects (useful for programmatic
    regression checks — e.g., assert that every expected_gate=True case passes
    the moonshot gate across all configs).
    """
    run_timestamp = datetime.now(timezone.utc).isoformat()
    log = logger.bind(benchmark_run_timestamp=run_timestamp)

    # Load test cases
    if not _TEST_CASES_PATH.exists():
        log.error("test_cases_file_not_found", path=str(_TEST_CASES_PATH))
        sys.exit(1)

    with _TEST_CASES_PATH.open(encoding="utf-8") as fh:
        test_cases: list[dict[str, Any]] = json.load(fh)

    log.info(
        "benchmark_sweep_start",
        test_case_count=len(test_cases),
        config_count=len(CONFIGS),
        total_runs=len(test_cases) * len(CONFIGS),
    )

    all_results: list[BenchmarkResult] = []

    for config in CONFIGS:
        for test_case in test_cases:
            # Set env vars before any import of the graph (agents read at import time)
            os.environ["AGENT_MODEL"] = config["agent_model"]
            os.environ["JUDGE_MODEL"] = config["judge_model"]

            try:
                result = _run_single(test_case, config)
            except Exception as exc:
                # Defensive catch-all: _run_single already catches most errors,
                # but this ensures one truly unexpected failure cannot abort the sweep.
                log.exception(
                    "unexpected_run_failure",
                    config=config["name"],
                    test_case=test_case["id"],
                    error=str(exc),
                )
                result = _error_result(
                    test_case["id"], config,
                    str(uuid.uuid4()),
                    datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )

            all_results.append(result)

    # Print comparison table
    _print_table(all_results)

    # Persist JSON results
    output_path = _save_results(all_results, run_timestamp)
    print(f"Results saved to: {output_path}")

    # Gate conformance check: warn when actual moonshot gate differs from expected
    print()
    print("Gate conformance (expected vs actual):")
    gate_header = f"  {'Test Case':<28} {'Expected':>10} {'Actual':>10} {'Match':>8}"
    print(gate_header)
    print("  " + "-" * (len(gate_header) - 2))

    expected_map = {tc["id"]: tc.get("expected_gate", None) for tc in test_cases}

    # Aggregate per test_case_id: any config passing the gate counts as "any passed"
    # because gate is a property of the idea, not the model config.
    gate_actual: dict[str, bool] = {}
    for r in all_results:
        if r.test_case_id not in gate_actual:
            gate_actual[r.test_case_id] = r.moonshot_gate_passed
        else:
            # If any config passes, treat as passed (the gate logic is deterministic
            # on the idea, so disagreement across configs indicates a flaky judge)
            gate_actual[r.test_case_id] = gate_actual[r.test_case_id] or r.moonshot_gate_passed

    for tc_id, expected in expected_map.items():
        actual = gate_actual.get(tc_id, False)
        match = "OK" if expected is None or (expected == actual) else "MISMATCH"
        expected_str = str(expected) if expected is not None else "N/A"
        print(f"  {tc_id:<28} {expected_str:>10} {str(actual):>10} {match:>8}")

    print()
    log.info("benchmark_sweep_complete", total_runs=len(all_results))
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark()
