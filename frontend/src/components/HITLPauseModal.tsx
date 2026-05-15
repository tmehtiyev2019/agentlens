"use client";

import { useState } from "react";
import AgentOutputCard from "./AgentOutputCard";

type HumanDecision = "approved" | "rejected";

interface HITLPauseModalProps {
  allOutputs: Record<string, unknown>;
  onResume: (decision: HumanDecision, comment?: string) => void;
  isSubmitting: boolean;
}

const TABS = [
  { id: "technical", label: "Technical" },
  { id: "market", label: "Market" },
  { id: "risk", label: "Risk" },
  { id: "cost", label: "Cost" },
  { id: "rag", label: "RAG" },
] as const;

const MAX_COMMENT_CHARS = 500;

export default function HITLPauseModal({
  allOutputs,
  onResume,
  isSubmitting,
}: HITLPauseModalProps) {
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]["id"]>("technical");
  const [comment, setComment] = useState("");

  const handleDecision = (decision: HumanDecision) => {
    if (isSubmitting) return;
    onResume(decision, comment.trim() || undefined);
  };

  const tabOutput = allOutputs[activeTab] as Record<string, unknown> | null | undefined;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col rounded-2xl border border-purple-700 bg-slate-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-slate-700 px-6 py-4">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-700">
            <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </span>
          <div>
            <h2 className="text-base font-semibold text-white">Review Analysis Before Kill Shot</h2>
            <p className="text-xs text-slate-400">
              All specialist agents have completed. Review outputs and decide whether to proceed.
            </p>
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-slate-700 px-6">
          {TABS.map((tab) => {
            const hasOutput = Boolean(allOutputs[tab.id]);
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "border-purple-500 text-purple-300"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab.label}
                {!hasOutput && (
                  <span className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-slate-600" />
                )}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <AgentOutputCard
            title={`${TABS.find((t) => t.id === activeTab)?.label ?? activeTab} Analysis`}
            output={tabOutput ?? null}
            isLoading={false}
          />
        </div>

        {/* Comment + actions */}
        <div className="border-t border-slate-700 px-6 py-4">
          <div className="mb-3">
            <label className="mb-1.5 block text-xs font-medium text-slate-400">
              Expert comment (optional)
            </label>
            <textarea
              value={comment}
              onChange={(e) => {
                if (e.target.value.length <= MAX_COMMENT_CHARS) {
                  setComment(e.target.value);
                }
              }}
              rows={3}
              placeholder="Add context or corrections..."
              disabled={isSubmitting}
              className="w-full resize-none rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:border-purple-500 focus:outline-none disabled:opacity-50"
            />
            <p className="mt-1 text-right text-xs text-slate-500">
              {comment.length}/{MAX_COMMENT_CHARS}
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => handleDecision("rejected")}
              disabled={isSubmitting}
              className="flex-1 rounded-lg border border-red-600 px-4 py-2.5 text-sm font-semibold text-red-400 transition-colors hover:bg-red-950 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSubmitting ? "Submitting..." : "Reject"}
            </button>
            <button
              onClick={() => handleDecision("approved")}
              disabled={isSubmitting}
              className="flex-1 rounded-lg bg-green-700 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-green-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isSubmitting ? "Submitting..." : "Approve & Continue →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
