"use client";

import { useState } from "react";

interface KillShotExperimentOutput {
  critical_assumption: string;
  why_this_assumption: string;
  experiment_description: string;
  success_criteria: string;
  failure_criteria: string;
  estimated_cost_usd: number;
  estimated_duration_weeks: number;
  required_resources: string[];
  informed_by_portfolio: boolean;
  portfolio_reference?: string | null;
  assumption_list: string[];
  confidence_level: "low" | "medium" | "high";
}

interface KillShotCardProps {
  output: KillShotExperimentOutput | null;
  isLoading: boolean;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-700 text-green-100",
  medium: "bg-yellow-700 text-yellow-100",
  low: "bg-red-800 text-red-100",
};

function formatCost(usd: number): string {
  if (usd >= 1_000_000) {
    return `$${(usd / 1_000_000).toFixed(1)}M`;
  }
  if (usd >= 1_000) {
    return `$${usd.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  }
  return `$${usd.toFixed(0)}`;
}

export default function KillShotCard({ output, isLoading }: KillShotCardProps) {
  const [assumptionsOpen, setAssumptionsOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-4 text-base font-semibold text-white">Kill Shot Experiment</h3>
        <div className="flex items-center gap-3 text-slate-400">
          <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-sm">Designing kill shot experiment...</span>
        </div>
      </div>
    );
  }

  if (!output) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-2 text-base font-semibold text-white">Kill Shot Experiment</h3>
        <p className="text-sm text-slate-500">No output available yet.</p>
      </div>
    );
  }

  const confidenceCls = CONFIDENCE_COLORS[output.confidence_level] ?? "bg-gray-700 text-gray-100";

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
      {/* Header */}
      <div className="mb-5 flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white">Kill Shot Experiment</h3>
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${confidenceCls}`}>
          {output.confidence_level} confidence
        </span>
      </div>

      {/* Critical assumption */}
      <div className="mb-4 rounded-lg border border-red-800 bg-red-950 px-4 py-3">
        <p className="mb-1 text-xs font-bold uppercase tracking-wide text-red-400">
          Critical Assumption
        </p>
        <p className="text-sm font-medium text-red-100">{output.critical_assumption}</p>
      </div>

      {/* Why this assumption */}
      <div className="mb-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Why this assumption?
        </p>
        <p className="text-sm text-slate-300">{output.why_this_assumption}</p>
      </div>

      {/* Experiment */}
      <div className="mb-4">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
          Experiment
        </p>
        <p className="text-sm text-slate-300">{output.experiment_description}</p>
      </div>

      {/* Success / Failure columns */}
      <div className="mb-4 grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-green-800 bg-green-950 px-3 py-2.5">
          <p className="mb-1 text-xs font-bold uppercase tracking-wide text-green-400">
            Success Criteria
          </p>
          <p className="text-sm text-green-100">{output.success_criteria}</p>
        </div>
        <div className="rounded-lg border border-red-800 bg-red-950 px-3 py-2.5">
          <p className="mb-1 text-xs font-bold uppercase tracking-wide text-red-400">
            Failure Criteria
          </p>
          <p className="text-sm text-red-100">{output.failure_criteria}</p>
        </div>
      </div>

      {/* Cost + duration */}
      <div className="mb-4 flex items-center gap-6 rounded-lg border border-slate-600 bg-slate-900 px-4 py-2.5">
        <div>
          <p className="text-xs text-slate-400">Est. Cost</p>
          <p className="text-lg font-bold text-white">{formatCost(output.estimated_cost_usd)}</p>
        </div>
        <div className="h-8 w-px bg-slate-700" />
        <div>
          <p className="text-xs text-slate-400">Duration</p>
          <p className="text-lg font-bold text-white">
            {output.estimated_duration_weeks}{" "}
            <span className="text-sm font-normal text-slate-300">weeks</span>
          </p>
        </div>
      </div>

      {/* Required resources */}
      {output.required_resources.length > 0 && (
        <div className="mb-4">
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Required Resources
          </p>
          <ul className="space-y-1">
            {output.required_resources.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-slate-500" />
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Portfolio reference */}
      {output.portfolio_reference && (
        <p className="mb-4 text-xs italic text-slate-400">
          Based on: {output.portfolio_reference}
        </p>
      )}

      {/* Assumptions collapsible */}
      {output.assumption_list.length > 0 && (
        <div className="border-t border-slate-700 pt-3">
          <button
            onClick={() => setAssumptionsOpen((o) => !o)}
            className="flex w-full items-center justify-between text-sm font-medium text-slate-300 hover:text-white"
          >
            <span>Assumptions ({output.assumption_list.length})</span>
            <span className="text-slate-400">{assumptionsOpen ? "▲" : "▼"}</span>
          </button>
          {assumptionsOpen && (
            <ul className="mt-2 space-y-1.5 pl-2">
              {output.assumption_list.map((a, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                  <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-slate-500" />
                  {a}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
