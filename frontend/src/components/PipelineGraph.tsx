"use client";

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeProps,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import NodeStatusBadge from "./NodeStatusBadge";
import type { NodeStatus } from "@/lib/api";

export const NODE_LABELS: Record<string, string> = {
  intent_classifier: "Intent Check",
  moonshot_evaluator: "Moonshot Gate",
  technical: "Technical",
  market: "Market",
  risk: "Risk",
  cost: "Cost",
  rag: "RAG",
  human_review: "Human Review",
  kill_shot: "Kill Shot",
  rejection_report: "Rejected",
};

interface AgentNodeData extends Record<string, unknown> {
  nodeId: string;
  label: string;
  status: NodeStatus;
  onNodeClick: (nodeId: string) => void;
}

function AgentNode({ data }: NodeProps<AgentNodeData>) {
  const { nodeId, label, status, onNodeClick } = data;

  const borderColor =
    status === "running"
      ? "border-blue-500"
      : status === "complete"
        ? "border-green-500"
        : status === "failed"
          ? "border-red-500"
          : status === "hitl_pause"
            ? "border-purple-500"
            : "border-gray-600";

  return (
    <button
      onClick={() => onNodeClick(nodeId)}
      className={`flex min-w-[120px] flex-col items-center gap-1.5 rounded-lg border-2 bg-slate-800 px-3 py-2 text-left shadow-lg transition-colors hover:bg-slate-700 ${borderColor}`}
    >
      <span className="text-sm font-semibold text-white">{label}</span>
      <NodeStatusBadge status={status} />
    </button>
  );
}

function RejectionNode({ data }: NodeProps<AgentNodeData>) {
  const { nodeId, label, status, onNodeClick } = data;

  return (
    <button
      onClick={() => onNodeClick(nodeId)}
      className="flex min-w-[100px] flex-col items-center gap-1.5 rounded-lg border-2 border-red-700 bg-red-950 px-3 py-2 shadow-lg transition-colors hover:bg-red-900"
    >
      <span className="text-sm font-semibold text-red-300">{label}</span>
      <NodeStatusBadge status={status} />
    </button>
  );
}

const nodeTypes = {
  agentNode: AgentNode,
  rejectionNode: RejectionNode,
};

interface PipelineGraphProps {
  nodeStates: Record<string, NodeStatus>;
  onNodeClick: (nodeId: string) => void;
}

function buildNodesAndEdges(
  nodeStates: Record<string, NodeStatus>,
  onNodeClick: (nodeId: string) => void,
): { nodes: Node[]; edges: Edge[] } {
  const getStatus = (id: string): NodeStatus => nodeStates[id] ?? "pending";

  const parallelIds = ["technical", "market", "risk", "cost", "rag"];
  const parallelSpacing = 160;
  const parallelStartX = -(((parallelIds.length - 1) * parallelSpacing) / 2);

  const nodes: Node[] = [
    {
      id: "intent_classifier",
      type: "agentNode",
      position: { x: 0, y: 0 },
      data: {
        nodeId: "intent_classifier",
        label: NODE_LABELS.intent_classifier,
        status: getStatus("intent_classifier"),
        onNodeClick,
      },
    },
    {
      id: "moonshot_evaluator",
      type: "agentNode",
      position: { x: 0, y: 120 },
      data: {
        nodeId: "moonshot_evaluator",
        label: NODE_LABELS.moonshot_evaluator,
        status: getStatus("moonshot_evaluator"),
        onNodeClick,
      },
    },
    ...parallelIds.map((id, i) => ({
      id,
      type: "agentNode" as const,
      position: { x: parallelStartX + i * parallelSpacing, y: 260 },
      data: {
        nodeId: id,
        label: NODE_LABELS[id],
        status: getStatus(id),
        onNodeClick,
      },
    })),
    {
      id: "human_review",
      type: "agentNode",
      position: { x: 0, y: 400 },
      data: {
        nodeId: "human_review",
        label: NODE_LABELS.human_review,
        status: getStatus("human_review"),
        onNodeClick,
      },
    },
    {
      id: "kill_shot",
      type: "agentNode",
      position: { x: 0, y: 520 },
      data: {
        nodeId: "kill_shot",
        label: NODE_LABELS.kill_shot,
        status: getStatus("kill_shot"),
        onNodeClick,
      },
    },
    {
      id: "rejection_report",
      type: "rejectionNode",
      position: { x: 420, y: 200 },
      data: {
        nodeId: "rejection_report",
        label: NODE_LABELS.rejection_report,
        status: getStatus("rejection_report"),
        onNodeClick,
      },
    },
  ];

  const edges: Edge[] = [
    // Main path
    {
      id: "e-intent-moonshot",
      source: "intent_classifier",
      target: "moonshot_evaluator",
      style: { stroke: "#94a3b8" },
    },
    // moonshot → each parallel agent
    ...parallelIds.map((id) => ({
      id: `e-moonshot-${id}`,
      source: "moonshot_evaluator",
      target: id,
      style: { stroke: "#94a3b8" },
    })),
    // parallel agents → human_review
    ...parallelIds.map((id) => ({
      id: `e-${id}-humanreview`,
      source: id,
      target: "human_review",
      style: { stroke: "#94a3b8" },
    })),
    {
      id: "e-humanreview-killshot",
      source: "human_review",
      target: "kill_shot",
      style: { stroke: "#94a3b8" },
    },
    // Rejection edges (dashed red)
    {
      id: "e-intent-rejected",
      source: "intent_classifier",
      target: "rejection_report",
      style: { stroke: "#ef4444", strokeDasharray: "5 4" },
      label: "reject",
      labelStyle: { fill: "#ef4444", fontSize: 10 },
    },
    {
      id: "e-moonshot-rejected",
      source: "moonshot_evaluator",
      target: "rejection_report",
      style: { stroke: "#ef4444", strokeDasharray: "5 4" },
      label: "fail gate",
      labelStyle: { fill: "#ef4444", fontSize: 10 },
    },
    {
      id: "e-humanreview-rejected",
      source: "human_review",
      target: "rejection_report",
      style: { stroke: "#ef4444", strokeDasharray: "5 4" },
      label: "reject",
      labelStyle: { fill: "#ef4444", fontSize: 10 },
    },
  ];

  return { nodes, edges };
}

export default function PipelineGraph({ nodeStates, onNodeClick }: PipelineGraphProps) {
  const stableOnNodeClick = useCallback(onNodeClick, [onNodeClick]);

  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => buildNodesAndEdges(nodeStates, stableOnNodeClick),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  // Sync status updates into node data without rebuilding edges
  useMemo(() => {
    setNodes((prev) =>
      prev.map((n) => {
        const status = nodeStates[n.id] ?? "pending";
        if ((n.data as AgentNodeData).status === status) return n;
        return {
          ...n,
          data: { ...(n.data as AgentNodeData), status, onNodeClick: stableOnNodeClick },
        };
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeStates]);

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        colorMode="dark"
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
      >
        <Background color="#334155" variant={BackgroundVariant.Dots} gap={20} size={1} />
        <Controls showInteractive={false} className="[&_button]:bg-slate-700 [&_button]:border-slate-600" />
      </ReactFlow>
    </div>
  );
}
