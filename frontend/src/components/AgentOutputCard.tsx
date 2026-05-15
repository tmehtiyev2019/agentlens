"use client";

import { useState } from "react";

interface AgentOutputCardProps {
  title: string;
  output: Record<string, unknown> | null;
  isLoading: boolean;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-700 text-green-100",
  medium: "bg-yellow-700 text-yellow-100",
  low: "bg-red-800 text-red-100",
};

function ConfidenceBadge({ level }: { level: string }) {
  const cls = CONFIDENCE_COLORS[level] ?? "bg-gray-700 text-gray-100";
  return (
    <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${cls}`}>
      {level} confidence
    </span>
  );
}

function CollapsibleSection({
  title,
  items,
}: {
  title: string;
  items: unknown[];
}) {
  const [open, setOpen] = useState(false);

  if (items.length === 0) return null;

  return (
    <div className="mt-3 border-t border-slate-700 pt-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-sm font-medium text-slate-300 hover:text-white"
      >
        <span>{title}</span>
        <span className="ml-2 text-slate-400">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <ul className="mt-2 space-y-1.5 pl-2">
          {items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
              <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-slate-500" />
              <span>{String(item)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function isScalar(value: unknown): boolean {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function formatValue(value: unknown): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (value > 999) return value.toLocaleString();
    return String(value);
  }
  return String(value);
}

function formatKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

const SKIP_KEYS = new Set([
  "confidence_level",
  "assumption_list",
  "evidence_citations",
]);

export default function AgentOutputCard({ title, output, isLoading }: AgentOutputCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-4 text-base font-semibold text-white">{title}</h3>
        <div className="flex items-center gap-3 text-slate-400">
          <svg
            className="h-5 w-5 animate-spin"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-sm">Running analysis...</span>
        </div>
      </div>
    );
  }

  if (!output) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-2 text-base font-semibold text-white">{title}</h3>
        <p className="text-sm text-slate-500">No output available yet.</p>
      </div>
    );
  }

  const confidenceLevel =
    typeof output.confidence_level === "string" ? output.confidence_level : null;

  const assumptionList = Array.isArray(output.assumption_list)
    ? (output.assumption_list as unknown[])
    : [];

  const evidenceCitations = Array.isArray(output.evidence_citations)
    ? (output.evidence_citations as unknown[])
    : [];

  const scalarEntries = Object.entries(output).filter(
    ([key, value]) => !SKIP_KEYS.has(key) && isScalar(value),
  );

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white">{title}</h3>
        {confidenceLevel && <ConfidenceBadge level={confidenceLevel} />}
      </div>

      {scalarEntries.length > 0 && (
        <dl className="space-y-2">
          {scalarEntries.map(([key, value]) => (
            <div key={key} className="flex gap-2 text-sm">
              <dt className="w-48 flex-shrink-0 text-slate-400">
                {formatKey(key)}
              </dt>
              <dd className="text-slate-200">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      <CollapsibleSection title={`Assumptions (${assumptionList.length})`} items={assumptionList} />
      <CollapsibleSection title={`Evidence Citations (${evidenceCitations.length})`} items={evidenceCitations} />
    </div>
  );
}
