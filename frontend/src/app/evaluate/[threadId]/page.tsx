"use client";

import { useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { usePipelineStream } from "@/hooks/usePipelineStream";
import { useHITLResume } from "@/hooks/useHITLResume";
import PipelineGraph from "@/components/PipelineGraph";
import OutputPanel from "@/components/OutputPanel";
import HITLPauseModal from "@/components/HITLPauseModal";
import { api } from "@/lib/api";

export default function EvaluatePage() {
  const params = useParams();
  const threadId = typeof params.threadId === "string" ? decodeURIComponent(params.threadId) : null;

  const { nodeStates, outputs, pipelineStatus, isHITLPaused, allOutputs } =
    usePipelineStream(threadId);

  const { resume, isSubmitting } = useHITLResume(threadId ?? "");

  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(nodeId);
  }, []);

  const handleHITLResume = async (
    decision: "approved" | "rejected",
    comment?: string,
  ) => {
    await resume(decision, comment);
  };

  const handleDownloadReport = async () => {
    if (!threadId) return;
    setIsDownloading(true);
    setDownloadError(null);

    try {
      const report = await api.report(threadId);
      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `agentlens-report-${threadId.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(
        err instanceof Error ? err.message : "Download failed.",
      );
    } finally {
      setIsDownloading(false);
    }
  };

  const isPipelineRejected = pipelineStatus === "rejected";
  const isPipelineComplete = pipelineStatus === "complete";

  if (!threadId) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-red-400">Invalid thread ID.</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#0F172A]">
      {/* Top bar */}
      <header className="flex flex-shrink-0 items-center justify-between border-b border-slate-700 bg-slate-900 px-5 py-3">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-sm font-semibold text-slate-300 hover:text-white">
            AgentLens
          </Link>
          <span className="text-slate-600">/</span>
          <span className="max-w-[200px] truncate text-xs text-slate-500 font-mono">{threadId}</span>
        </div>

        <div className="flex items-center gap-3">
          {/* Pipeline status pill */}
          {pipelineStatus && (
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                isPipelineComplete
                  ? "bg-green-800 text-green-200"
                  : isPipelineRejected
                    ? "bg-red-800 text-red-200"
                    : isHITLPaused
                      ? "bg-purple-800 text-purple-200"
                      : pipelineStatus === "error"
                        ? "bg-red-800 text-red-200"
                        : "bg-blue-800 text-blue-200"
              }`}
            >
              {isPipelineComplete
                ? "Complete"
                : isPipelineRejected
                  ? "Rejected"
                  : isHITLPaused
                    ? "Awaiting Review"
                    : pipelineStatus === "error"
                      ? "Error"
                      : "Running"}
            </span>
          )}
        </div>
      </header>

      {/* Status banners */}
      {isPipelineComplete && (
        <div className="flex flex-shrink-0 items-center justify-between border-b border-green-800 bg-green-950 px-5 py-2.5">
          <div className="flex items-center gap-2">
            <svg className="h-4 w-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-sm font-medium text-green-300">Analysis Complete</span>
          </div>
          <button
            onClick={handleDownloadReport}
            disabled={isDownloading}
            className="rounded-lg bg-green-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-green-600 disabled:opacity-50"
          >
            {isDownloading ? "Downloading..." : "Download JSON Report"}
          </button>
          {downloadError && (
            <span className="ml-3 text-xs text-red-400">{downloadError}</span>
          )}
        </div>
      )}

      {isPipelineRejected && (
        <div className="flex flex-shrink-0 items-center gap-2 border-b border-red-800 bg-red-950 px-5 py-2.5">
          <svg className="h-4 w-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
          <span className="text-sm font-medium text-red-300">
            Pipeline rejected.
          </span>
          {outputs.rejection_report && typeof outputs.rejection_report === "object" && (
            <span className="text-sm text-red-400">
              {String(
                (outputs.rejection_report as Record<string, unknown>).reason ??
                  (outputs.rejection_report as Record<string, unknown>).gate_failure_reason ??
                  "",
              )}
            </span>
          )}
        </div>
      )}

      {/* Main content */}
      <div className="flex min-h-0 flex-1">
        {/* Left: Pipeline graph */}
        <div className="flex w-1/2 flex-col border-r border-slate-700">
          <div className="flex-shrink-0 border-b border-slate-700 px-4 py-2">
            <p className="text-xs font-medium text-slate-400">Pipeline Graph</p>
          </div>
          <div className="flex-1 overflow-hidden">
            <PipelineGraph
              nodeStates={nodeStates}
              onNodeClick={handleNodeClick}
            />
          </div>
        </div>

        {/* Right: Output panel */}
        <div className="flex w-1/2 flex-col">
          <div className="flex-shrink-0 border-b border-slate-700 px-4 py-2">
            <p className="text-xs font-medium text-slate-400">
              {selectedNode
                ? `Output — ${selectedNode.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}`
                : "Node Output"}
            </p>
          </div>
          <div className="flex-1 overflow-hidden">
            <OutputPanel
              selectedNode={selectedNode}
              outputs={outputs}
              nodeStates={nodeStates}
            />
          </div>
        </div>
      </div>

      {/* HITL pause modal */}
      {isHITLPaused && (
        <HITLPauseModal
          allOutputs={allOutputs}
          onResume={handleHITLResume}
          isSubmitting={isSubmitting}
        />
      )}
    </div>
  );
}
