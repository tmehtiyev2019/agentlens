"use client";

import type { NodeStatus } from "@/lib/api";

interface NodeStatusBadgeProps {
  status: NodeStatus;
}

const STATUS_CONFIG: Record<
  NodeStatus,
  { label: string; className: string; extraClassName?: string }
> = {
  pending: {
    label: "Pending",
    className: "bg-gray-600 text-gray-200",
  },
  running: {
    label: "Running",
    className: "bg-blue-600 text-white node-running-pulse",
  },
  complete: {
    label: "Complete",
    className: "bg-green-600 text-white",
  },
  failed: {
    label: "Failed",
    className: "bg-red-600 text-white",
  },
  hitl_pause: {
    label: "Paused",
    className: "bg-purple-600 text-white",
  },
};

export default function NodeStatusBadge({ status }: NodeStatusBadgeProps) {
  const config = STATUS_CONFIG[status];

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${config.className}`}
    >
      {config.label}
    </span>
  );
}
