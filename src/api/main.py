"""
FastAPI application serving the AgentLens TEA pipeline via HTTP.

Supports:
  - POST /evaluate              — submit idea, returns thread_id immediately
  - GET  /evaluate/{id}/stream  — SSE stream of PipelineEvent objects
  - POST /evaluate/{id}/resume  — inject HITL decision and continue
  - GET  /evaluate/{id}/status  — current pipeline status (for reconnect)
  - GET  /evaluate/{id}/report  — final report once complete
  - GET  /benchmark             — trigger benchmark run (async)
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AgentLens TEA Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LangGraph graph — compiled once at startup with SqliteSaver checkpointer
# ---------------------------------------------------------------------------

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agents.orchestrator import build_graph

_conn = sqlite3.connect(
    os.getenv("CHECKPOINT_DB", "./checkpoints.db"),
    check_same_thread=False,
)
_checkpointer = SqliteSaver(_conn)
graph = build_graph(checkpointer=_checkpointer)

# ---------------------------------------------------------------------------
# In-process state — keyed by thread_id
# ---------------------------------------------------------------------------

# asyncio.Queue per thread_id — SSE generator reads from this
event_queues: dict[str, asyncio.Queue] = {}

# Latest known state snapshot — populated on every streamed chunk
pipeline_states: dict[str, dict] = {}

# Thread pool for running blocking graph.stream() calls off the event loop
_executor = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    idea: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    thread_id: str
    status: str  # "started"


class PipelineEvent(BaseModel):
    type: Literal[
        "node_started",
        "node_completed",
        "node_failed",
        "hitl_pause",
        "pipeline_complete",
        "pipeline_rejected",
    ]
    node: str | None = None
    output: dict | None = None
    reason: str | None = None
    thread_id: str


class ResumeRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: str | None = None


class ResumeResponse(BaseModel):
    thread_id: str
    status: str  # "resumed" | "error"


class StatusResponse(BaseModel):
    thread_id: str
    status: Literal["running", "hitl_pause", "complete", "rejected", "error"]
    current_node: str | None = None
    completed_nodes: list[str]


class BenchmarkResponse(BaseModel):
    run_id: str
    status: str  # "started"


# ---------------------------------------------------------------------------
# Pipeline runner helpers
# ---------------------------------------------------------------------------

def _serialize(value: Any) -> Any:
    """Recursively make a value JSON-serialisable (Pydantic models → dicts)."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(i) for i in value]
    return value


def _stream_graph(thread_id: str, initial_state: dict, loop: asyncio.AbstractEventLoop) -> None:
    """
    Run graph.stream() in a worker thread. For each node output, build a
    PipelineEvent and enqueue it on the asyncio event loop that owns the queue.

    The sentinel value None is enqueued when the stream ends (success or error).
    """
    config = {"configurable": {"thread_id": thread_id}}
    log = logger.bind(thread_id=thread_id)

    # Track which nodes have completed so far
    completed: list[str] = []

    try:
        for chunk in graph.stream(initial_state, config=config):
            # chunk is {node_name: state_update_dict}
            for node_name, state_update in chunk.items():
                # Merge update into the latest known snapshot
                snapshot = pipeline_states.get(thread_id, {})
                snapshot.update(state_update if isinstance(state_update, dict) else {})
                pipeline_states[thread_id] = snapshot

                completed.append(node_name)

                # Determine whether this node signals a failure
                errors = state_update.get("errors", []) if isinstance(state_update, dict) else []
                has_error = bool(errors)

                if node_name == "human_review":
                    # Graph is about to pause — emit HITL pause event before it returns
                    _enqueue(
                        loop,
                        thread_id,
                        PipelineEvent(
                            type="hitl_pause",
                            node=node_name,
                            output=_serialize(state_update) if isinstance(state_update, dict) else None,
                            thread_id=thread_id,
                        ),
                    )
                    pipeline_states[thread_id]["_status"] = "hitl_pause"
                    pipeline_states[thread_id]["_completed_nodes"] = completed[:]
                    log.info("pipeline_hitl_pause", node=node_name)
                    return  # LangGraph will have suspended; stream ends here naturally

                event_type: Literal[
                    "node_started",
                    "node_completed",
                    "node_failed",
                    "hitl_pause",
                    "pipeline_complete",
                    "pipeline_rejected",
                ] = "node_failed" if has_error else "node_completed"

                _enqueue(
                    loop,
                    thread_id,
                    PipelineEvent(
                        type=event_type,
                        node=node_name,
                        output=_serialize(state_update) if isinstance(state_update, dict) else None,
                        reason=errors[0] if errors else None,
                        thread_id=thread_id,
                    ),
                )

                log.info("node_completed", node=node_name, has_error=has_error)

        # Stream finished without a HITL pause — check final state
        final = pipeline_states.get(thread_id, {})
        completed_nodes = completed

        # Determine terminal outcome from state
        rejection_signals = (
            final.get("intent_safe") is False
            or (
                final.get("moonshot_evaluation") is not None
                and not final["moonshot_evaluation"].passes_moonshot_gate
                if hasattr(final.get("moonshot_evaluation"), "passes_moonshot_gate")
                else False
            )
            or final.get("human_decision") == "rejected"
        )

        if rejection_signals:
            terminal_type: Literal["pipeline_complete", "pipeline_rejected"] = "pipeline_rejected"
            pipeline_states[thread_id]["_status"] = "rejected"
        else:
            terminal_type = "pipeline_complete"
            pipeline_states[thread_id]["_status"] = "complete"

        pipeline_states[thread_id]["_completed_nodes"] = completed_nodes

        _enqueue(
            loop,
            thread_id,
            PipelineEvent(
                type=terminal_type,
                node=None,
                output=_serialize({k: v for k, v in final.items() if not k.startswith("_")}),
                thread_id=thread_id,
            ),
        )
        log.info("pipeline_finished", status=terminal_type)

    except Exception as exc:
        log.error("pipeline_error", error=str(exc))
        pipeline_states.setdefault(thread_id, {})["_status"] = "error"
        _enqueue(
            loop,
            thread_id,
            PipelineEvent(
                type="pipeline_rejected",
                node=None,
                reason=str(exc),
                thread_id=thread_id,
            ),
        )

    finally:
        # Sentinel: tell the SSE generator the stream is over
        _enqueue(loop, thread_id, None)


def _enqueue(loop: asyncio.AbstractEventLoop, thread_id: str, event: PipelineEvent | None) -> None:
    """Thread-safe enqueue into the asyncio.Queue owned by the event loop."""
    queue = event_queues.get(thread_id)
    if queue is None:
        return
    asyncio.run_coroutine_threadsafe(queue.put(event), loop)


async def _launch_pipeline(thread_id: str, initial_state: dict) -> None:
    """Start graph.stream() in a thread pool worker, forwarding the running loop."""
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _stream_graph, thread_id, initial_state, loop)


def _stream_graph_resume(thread_id: str, loop: asyncio.AbstractEventLoop) -> None:
    """
    Resume variant: graph.stream() with no initial_state so it loads from checkpoint.
    """
    config = {"configurable": {"thread_id": thread_id}}
    log = logger.bind(thread_id=thread_id)
    completed: list[str] = pipeline_states.get(thread_id, {}).get("_completed_nodes", [])

    try:
        for chunk in graph.stream(None, config=config):
            for node_name, state_update in chunk.items():
                snapshot = pipeline_states.get(thread_id, {})
                snapshot.update(state_update if isinstance(state_update, dict) else {})
                pipeline_states[thread_id] = snapshot
                completed.append(node_name)

                errors = state_update.get("errors", []) if isinstance(state_update, dict) else []
                has_error = bool(errors)

                event_type: Literal[
                    "node_started",
                    "node_completed",
                    "node_failed",
                    "hitl_pause",
                    "pipeline_complete",
                    "pipeline_rejected",
                ] = "node_failed" if has_error else "node_completed"

                _enqueue(
                    loop,
                    thread_id,
                    PipelineEvent(
                        type=event_type,
                        node=node_name,
                        output=_serialize(state_update) if isinstance(state_update, dict) else None,
                        reason=errors[0] if errors else None,
                        thread_id=thread_id,
                    ),
                )
                log.info("node_completed_resumed", node=node_name, has_error=has_error)

        final = pipeline_states.get(thread_id, {})
        rejection_signals = final.get("human_decision") == "rejected" or final.get("intent_safe") is False

        if rejection_signals:
            terminal_type: Literal["pipeline_complete", "pipeline_rejected"] = "pipeline_rejected"
            pipeline_states[thread_id]["_status"] = "rejected"
        else:
            terminal_type = "pipeline_complete"
            pipeline_states[thread_id]["_status"] = "complete"

        pipeline_states[thread_id]["_completed_nodes"] = completed

        _enqueue(
            loop,
            thread_id,
            PipelineEvent(
                type=terminal_type,
                node=None,
                output=_serialize({k: v for k, v in final.items() if not k.startswith("_")}),
                thread_id=thread_id,
            ),
        )
        log.info("pipeline_resumed_finished", status=terminal_type)

    except Exception as exc:
        log.error("pipeline_resume_error", error=str(exc))
        pipeline_states.setdefault(thread_id, {})["_status"] = "error"
        _enqueue(
            loop,
            thread_id,
            PipelineEvent(
                type="pipeline_rejected",
                node=None,
                reason=str(exc),
                thread_id=thread_id,
            ),
        )

    finally:
        _enqueue(loop, thread_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest) -> EvaluateResponse:
    """Submit an idea. Returns thread_id immediately; pipeline runs in the background."""
    thread_id = str(uuid.uuid4())
    log = logger.bind(thread_id=thread_id)

    event_queues[thread_id] = asyncio.Queue()
    pipeline_states[thread_id] = {"_status": "running", "_completed_nodes": []}

    initial_state = {
        "idea": body.idea,
        "errors": [],
        "intent_safe": None,
        "moonshot_evaluation": None,
        "technical_output": None,
        "market_output": None,
        "risk_output": None,
        "cost_output": None,
        "rag_output": None,
        "human_decision": None,
        "human_comment": None,
        "kill_shot": None,
        "grounded": False,
    }

    asyncio.create_task(_launch_pipeline(thread_id, initial_state))
    log.info("pipeline_started")

    return EvaluateResponse(thread_id=thread_id, status="started")


@app.get("/evaluate/{thread_id}/stream")
async def stream_events(thread_id: str) -> StreamingResponse:
    """SSE stream of PipelineEvent objects. Blocks until the pipeline completes."""
    if thread_id not in event_queues:
        raise HTTPException(status_code=404, detail="thread_id not found")

    async def event_generator():
        queue = event_queues[thread_id]
        while True:
            event: PipelineEvent | None = await queue.get()
            if event is None:
                # Sentinel — stream is done; clean up the queue
                event_queues.pop(thread_id, None)
                break
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/evaluate/{thread_id}/resume", response_model=ResumeResponse)
async def resume(thread_id: str, body: ResumeRequest) -> ResumeResponse:
    """
    Inject HITL decision into a paused pipeline and restart streaming.

    graph.update_state() writes human_decision + human_comment into the
    persisted checkpoint; then graph.stream(None, ...) resumes from that point.
    """
    log = logger.bind(thread_id=thread_id)

    state = pipeline_states.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id not found")
    if state.get("_status") != "hitl_pause":
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is not paused — current status: {state.get('_status')}",
        )

    try:
        config = {"configurable": {"thread_id": thread_id}}
        graph.update_state(
            config=config,
            values={
                "human_decision": body.decision,
                "human_comment": body.comment,
            },
            # as_node tells LangGraph which node produced this update so routing works
            as_node="human_review",
        )
    except Exception as exc:
        log.error("hitl_resume_state_update_failed", error=str(exc))
        return ResumeResponse(thread_id=thread_id, status="error")

    # Allocate a fresh queue so the SSE /stream endpoint can be re-opened
    event_queues[thread_id] = asyncio.Queue()
    pipeline_states[thread_id]["_status"] = "running"

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _stream_graph_resume, thread_id, loop)

    log.info("pipeline_resumed", decision=body.decision)
    return ResumeResponse(thread_id=thread_id, status="resumed")


@app.get("/evaluate/{thread_id}/status", response_model=StatusResponse)
async def status(thread_id: str) -> StatusResponse:
    """Current pipeline status — useful for clients reconnecting after a disconnect."""
    state = pipeline_states.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id not found")

    raw_status = state.get("_status", "running")
    # Map internal status strings to the declared Literal values
    status_map: dict[str, Literal["running", "hitl_pause", "complete", "rejected", "error"]] = {
        "running": "running",
        "hitl_pause": "hitl_pause",
        "complete": "complete",
        "rejected": "rejected",
        "error": "error",
    }
    mapped_status = status_map.get(raw_status, "running")

    return StatusResponse(
        thread_id=thread_id,
        status=mapped_status,
        current_node=None,  # not tracked at this granularity; SSE carries per-node events
        completed_nodes=state.get("_completed_nodes", []),
    )


@app.get("/evaluate/{thread_id}/report")
async def report(thread_id: str) -> dict:
    """
    Return the final assembled report once the pipeline is complete.

    Raises 409 if the pipeline is still running or paused.
    """
    state = pipeline_states.get(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail="thread_id not found")

    current_status = state.get("_status")
    if current_status not in ("complete", "rejected"):
        raise HTTPException(
            status_code=409,
            detail=f"Report not yet available — pipeline status: {current_status}",
        )

    # Filter out internal tracking keys (prefixed with _)
    public_state = {k: _serialize(v) for k, v in state.items() if not k.startswith("_")}

    return {
        "thread_id": thread_id,
        "status": current_status,
        "report": public_state,
    }


# ---------------------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------------------


@app.get("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark() -> BenchmarkResponse:
    """
    Trigger a benchmark sweep in the background.

    The benchmark runner is imported lazily so the API starts even when
    benchmark dependencies (e.g. RAGAS, heavy models) are not fully ready.
    Returns a run_id immediately; progress can be tracked via logs.
    """
    run_id = str(uuid.uuid4())
    log = logger.bind(run_id=run_id)

    async def _run_benchmark_task():
        try:
            log.info("benchmark_started")
            # Import here to avoid pulling in heavy optional deps at startup
            from src.evaluation.benchmark import run_benchmark_suite  # type: ignore[import]
            await asyncio.get_running_loop().run_in_executor(
                _executor, run_benchmark_suite, run_id
            )
            log.info("benchmark_complete")
        except Exception as exc:
            log.error("benchmark_failed", error=str(exc))

    asyncio.create_task(_run_benchmark_task())
    return BenchmarkResponse(run_id=run_id, status="started")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}
