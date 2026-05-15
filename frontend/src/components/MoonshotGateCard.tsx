"use client";

interface MoonshotEvaluationOutput {
  problem_is_real: boolean;
  problem_explanation: string;
  solution_is_feasible: boolean;
  solution_explanation: string;
  technology_is_available: boolean;
  technology_trl: number;
  technology_explanation: string;
  passes_moonshot_gate: boolean;
  gate_failure_reason?: string | null;
  confidence_level: "low" | "medium" | "high";
}

interface MoonshotGateCardProps {
  output: MoonshotEvaluationOutput | null;
  isLoading: boolean;
}

function PassFailBadge({ pass }: { pass: boolean }) {
  return pass ? (
    <span className="inline-flex items-center gap-1 rounded-full bg-green-800 px-2.5 py-0.5 text-xs font-semibold text-green-200">
      <svg className="h-3 w-3" viewBox="0 0 12 12" fill="none">
        <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      PASS
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-full bg-red-800 px-2.5 py-0.5 text-xs font-semibold text-red-200">
      <svg className="h-3 w-3" viewBox="0 0 12 12" fill="none">
        <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
      FAIL
    </span>
  );
}

function TrlBadge({ trl }: { trl: number }) {
  const color =
    trl >= 7 ? "bg-green-800 text-green-200" :
    trl >= 4 ? "bg-yellow-800 text-yellow-200" :
    "bg-red-800 text-red-200";

  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${color}`}>
      TRL {trl}
    </span>
  );
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-700 text-green-100",
  medium: "bg-yellow-700 text-yellow-100",
  low: "bg-red-800 text-red-100",
};

export default function MoonshotGateCard({ output, isLoading }: MoonshotGateCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-4 text-base font-semibold text-white">Moonshot Gate</h3>
        <div className="flex items-center gap-3 text-slate-400">
          <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span className="text-sm">Evaluating moonshot criteria...</span>
        </div>
      </div>
    );
  }

  if (!output) {
    return (
      <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
        <h3 className="mb-2 text-base font-semibold text-white">Moonshot Gate</h3>
        <p className="text-sm text-slate-500">No output available yet.</p>
      </div>
    );
  }

  const confidenceCls = CONFIDENCE_COLORS[output.confidence_level] ?? "bg-gray-700 text-gray-100";

  const rows = [
    {
      q: "Q1 — Problem Reality",
      question: "Is this a real problem affecting millions of people?",
      pass: output.problem_is_real,
      explanation: output.problem_explanation,
      badge: null,
    },
    {
      q: "Q2 — Solution Feasibility",
      question: "Is there a feasible, radical (10x improvement) solution?",
      pass: output.solution_is_feasible,
      explanation: output.solution_explanation,
      badge: null,
    },
    {
      q: "Q3 — Technology Readiness",
      question: "Is the enabling technology at TRL 4+ or achievable within 5 years?",
      pass: output.technology_is_available,
      explanation: output.technology_explanation,
      badge: <TrlBadge trl={output.technology_trl} />,
    },
  ];

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white">Moonshot Gate</h3>
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${confidenceCls}`}>
          {output.confidence_level} confidence
        </span>
      </div>

      <div className="space-y-4">
        {rows.map((row) => (
          <div key={row.q} className="rounded-lg border border-slate-700 bg-slate-900 p-3">
            <div className="mb-1.5 flex flex-wrap items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wide text-slate-400">{row.q}</span>
              <PassFailBadge pass={row.pass} />
              {row.badge}
            </div>
            <p className="mb-1 text-xs text-slate-400">{row.question}</p>
            <p className="text-sm text-slate-200">{row.explanation}</p>
          </div>
        ))}
      </div>

      <div
        className={`mt-5 rounded-lg px-4 py-3 text-center text-sm font-bold uppercase tracking-widest ${
          output.passes_moonshot_gate
            ? "bg-green-900 text-green-300 border border-green-700"
            : "bg-red-900 text-red-300 border border-red-700"
        }`}
      >
        {output.passes_moonshot_gate
          ? "MOONSHOT CONFIRMED"
          : `NOT A MOONSHOT${output.gate_failure_reason ? `: ${output.gate_failure_reason}` : ""}`}
      </div>
    </div>
  );
}
