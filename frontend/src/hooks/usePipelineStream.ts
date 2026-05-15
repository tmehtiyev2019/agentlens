"use client";

import { useEffect, useReducer, useRef } from "react";
import { type NodeStatus, type PipelineStatus, type PipelineEvent, api } from "@/lib/api";

type NodeStates = Record<string, NodeStatus>;
type NodeOutputs = Record<string, Record<string, unknown>>;

interface StreamState {
  nodeStates: NodeStates;
  outputs: NodeOutputs;
  pipelineStatus: PipelineStatus | null;
  isHITLPaused: boolean;
}

type StreamAction =
  | { type: "NODE_STARTED"; node: string }
  | { type: "NODE_COMPLETED"; node: string; output: Record<string, unknown> | null }
  | { type: "NODE_FAILED"; node: string }
  | { type: "HITL_PAUSE"; node: string | null }
  | { type: "PIPELINE_COMPLETE" }
  | { type: "PIPELINE_REJECTED" }
  | { type: "RESET" };

const initialState: StreamState = {
  nodeStates: {},
  outputs: {},
  pipelineStatus: null,
  isHITLPaused: false,
};

function reducer(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case "NODE_STARTED":
      return {
        ...state,
        pipelineStatus: state.pipelineStatus === null ? "running" : state.pipelineStatus,
        nodeStates: { ...state.nodeStates, [action.node]: "running" },
      };

    case "NODE_COMPLETED": {
      const nextOutputs =
        action.output !== null
          ? { ...state.outputs, [action.node]: action.output }
          : state.outputs;
      return {
        ...state,
        nodeStates: { ...state.nodeStates, [action.node]: "complete" },
        outputs: nextOutputs,
      };
    }

    case "NODE_FAILED":
      return {
        ...state,
        nodeStates: { ...state.nodeStates, [action.node]: "failed" },
      };

    case "HITL_PAUSE": {
      const nextNodeStates =
        action.node !== null
          ? { ...state.nodeStates, [action.node]: "hitl_pause" as NodeStatus }
          : state.nodeStates;
      return {
        ...state,
        nodeStates: nextNodeStates,
        pipelineStatus: "hitl_pause",
        isHITLPaused: true,
      };
    }

    case "PIPELINE_COMPLETE":
      return { ...state, pipelineStatus: "complete", isHITLPaused: false };

    case "PIPELINE_REJECTED":
      return { ...state, pipelineStatus: "rejected", isHITLPaused: false };

    case "RESET":
      return initialState;

    default:
      return state;
  }
}

function parsePipelineEvent(raw: string): PipelineEvent | null {
  try {
    return JSON.parse(raw) as PipelineEvent;
  } catch {
    return null;
  }
}

export interface PipelineStreamResult {
  nodeStates: NodeStates;
  outputs: NodeOutputs;
  pipelineStatus: PipelineStatus | null;
  isHITLPaused: boolean;
  allOutputs: NodeOutputs;
}

export function usePipelineStream(threadId: string | null): PipelineStreamResult {
  const [state, dispatch] = useReducer(reducer, initialState);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (threadId === null) {
      dispatch({ type: "RESET" });
      return;
    }

    dispatch({ type: "RESET" });

    const url = api.streamUrl(threadId);
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (event: MessageEvent<string>) => {
      const parsed = parsePipelineEvent(event.data);
      if (parsed === null) return;

      switch (parsed.type) {
        case "node_started":
          if (parsed.node !== null) {
            dispatch({ type: "NODE_STARTED", node: parsed.node });
          }
          break;

        case "node_completed":
          if (parsed.node !== null) {
            dispatch({
              type: "NODE_COMPLETED",
              node: parsed.node,
              output: parsed.output,
            });
          }
          break;

        case "node_failed":
          if (parsed.node !== null) {
            dispatch({ type: "NODE_FAILED", node: parsed.node });
          }
          break;

        case "hitl_pause":
          dispatch({ type: "HITL_PAUSE", node: parsed.node });
          break;

        case "pipeline_complete":
          dispatch({ type: "PIPELINE_COMPLETE" });
          es.close();
          esRef.current = null;
          break;

        case "pipeline_rejected":
          dispatch({ type: "PIPELINE_REJECTED" });
          es.close();
          esRef.current = null;
          break;
      }
    };

    es.onerror = () => {
      // EventSource will attempt to reconnect automatically unless we close it.
      // Only close when the pipeline has reached a terminal state; otherwise let
      // the browser handle reconnection (e.g. after a brief server restart).
      if (
        state.pipelineStatus === "complete" ||
        state.pipelineStatus === "rejected" ||
        state.pipelineStatus === "error"
      ) {
        es.close();
        esRef.current = null;
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
    // threadId is the only reactive dependency; we intentionally re-open the
    // stream whenever it changes and do not close on pipelineStatus transitions
    // inside this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  return {
    nodeStates: state.nodeStates,
    outputs: state.outputs,
    pipelineStatus: state.pipelineStatus,
    isHITLPaused: state.isHITLPaused,
    allOutputs: state.outputs,
  };
}
