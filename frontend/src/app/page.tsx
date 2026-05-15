"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import IdeaInputForm from "@/components/IdeaInputForm";
import { api } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (idea: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const { thread_id } = await api.evaluate(idea);
      router.push(`/evaluate/${encodeURIComponent(thread_id)}`);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start analysis. Please try again.";
      setError(message);
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="mb-10 text-center">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-800 bg-blue-950 px-3 py-1 text-xs font-medium text-blue-300">
            <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
            Multi-Agent Analysis Pipeline
          </div>
          <h1 className="mb-3 text-4xl font-bold tracking-tight text-white">
            AgentLens
          </h1>
          <p className="text-lg text-slate-400">
            Moonshot TEA Evaluator
          </p>
          <p className="mt-2 text-sm text-slate-500">
            A panel of specialist agents evaluates your idea across technical feasibility,
            market opportunity, risk, cost, and knowledge retrieval — in parallel.
          </p>
        </div>

        {/* Form card */}
        <div className="rounded-2xl border border-slate-700 bg-slate-800/60 p-6 shadow-xl backdrop-blur-sm">
          <h2 className="mb-1 text-sm font-semibold text-slate-300">
            Describe your moonshot idea
          </h2>
          <p className="mb-4 text-xs text-slate-500">
            Be specific: problem, proposed solution, and enabling technology.
          </p>
          <IdeaInputForm onSubmit={handleSubmit} isLoading={isLoading} />

          {error && (
            <div className="mt-4 rounded-lg border border-red-700 bg-red-950 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>

        {/* Pipeline legend */}
        <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "Intent Check", desc: "Safety gate" },
            { label: "Moonshot Gate", desc: "Q1·Q2·Q3 criteria" },
            { label: "5 Parallel Agents", desc: "Tech·Market·Risk·Cost·RAG" },
            { label: "Kill Shot", desc: "Cheapest falsification test" },
          ].map((step) => (
            <div
              key={step.label}
              className="rounded-lg border border-slate-700 bg-slate-800/40 px-3 py-2 text-center"
            >
              <p className="text-xs font-semibold text-slate-300">{step.label}</p>
              <p className="mt-0.5 text-xs text-slate-500">{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
