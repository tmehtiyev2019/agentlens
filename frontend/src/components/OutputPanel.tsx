"use client";

import type { NodeStatus } from "@/lib/api";
import AgentOutputCard from "./AgentOutputCard";
import MoonshotGateCard from "./MoonshotGateCard";
import KillShotCard from "./KillShotCard";

interface OutputPanelProps {
  selectedNode: string | null;
  outputs: Record<string, unknown>;
  nodeStates: Record<string, NodeStatus>;
}

const AGENT_TITLES: Record<string, string> = {
  technical: "Technical Feasibility",
  market: "Market Opportunity",
  risk: "Risk Assessment",
  cost: "Cost Estimation",
  rag: "RAG Knowledge Retrieval",
};

export default function OutputPanel({ selectedNode, outputs, nodeStates }: OutputPanelProps) {
  if (!selectedNode) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="mb-3 text-4xl text-slate-600">
            <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-slate-500">Select a node to view output</p>
        </div>
      </div>
    );
  }

  const nodeStatus = nodeStates[selectedNode] ?? "pending";
  const isRunning = nodeStatus === "running";
  const isLoading = isRunning || nodeStatus === "pending";
  const rawOutput = outputs[selectedNode] as Record<string, unknown> | undefined;

  if (selectedNode === "moonshot_evaluator") {
    return (
      <div className="h-full overflow-y-auto p-4">
        <MoonshotGateCard
          output={(rawOutput as Parameters<typeof MoonshotGateCard>[0]["output"]) ?? null}
          isLoading={isLoading && !rawOutput}
        />
      </div>
    );
  }

  if (selectedNode === "kill_shot") {
    return (
      <div className="h-full overflow-y-auto p-4">
        <KillShotCard
          output={(rawOutput as Parameters<typeof KillShotCard>[0]["output"]) ?? null}
          isLoading={isLoading && !rawOutput}
        />
      </div>
    );
  }

  if (AGENT_TITLES[selectedNode]) {
    return (
      <div className="h-full overflow-y-auto p-4">
        <AgentOutputCard
          title={AGENT_TITLES[selectedNode]}
          output={rawOutput ?? null}
          isLoading={isLoading && !rawOutput}
        />
      </div>
    );
  }

  // intent_classifier, human_review, rejection_report, or any unknown node
  if (rawOutput) {
    const title =
      selectedNode === "intent_classifier"
        ? "Intent Check"
        : selectedNode === "human_review"
          ? "Human Review"
          : selectedNode === "rejection_report"
            ? "Rejection Report"
            : selectedNode;

    return (
      <div className="h-full overflow-y-auto p-4">
        <AgentOutputCard
          title={title}
          output={rawOutput}
          isLoading={false}
        />
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center p-4">
      <p className="text-sm text-slate-500">
        {isRunning
          ? `${selectedNode} is running...`
          : `No output available for ${selectedNode}`}
      </p>
    </div>
  );
}
