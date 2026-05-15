"use client";

import { useState } from "react";

interface IdeaInputFormProps {
  onSubmit: (idea: string) => void;
  isLoading: boolean;
}

const MAX_CHARS = 2000;
const MIN_CHARS = 50;

export default function IdeaInputForm({ onSubmit, isLoading }: IdeaInputFormProps) {
  const [idea, setIdea] = useState("");

  const charCount = idea.length;
  const isUnderMin = charCount < MIN_CHARS;
  const isDisabled = isLoading || isUnderMin || charCount === 0;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isDisabled) {
      onSubmit(idea.trim());
    }
  };

  const charCountColor =
    charCount > MAX_CHARS * 0.9
      ? "text-red-400"
      : charCount > MAX_CHARS * 0.75
        ? "text-yellow-400"
        : "text-slate-400";

  return (
    <form onSubmit={handleSubmit} className="flex w-full flex-col gap-4">
      <div className="relative">
        <textarea
          value={idea}
          onChange={(e) => {
            if (e.target.value.length <= MAX_CHARS) {
              setIdea(e.target.value);
            }
          }}
          rows={6}
          disabled={isLoading}
          placeholder="Describe your moonshot idea in detail. What problem does it solve? What is the proposed solution? What technology does it rely on? The more context you provide, the more accurate the analysis will be. (min 50 characters)"
          className="w-full resize-none rounded-xl border border-slate-600 bg-slate-800 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 shadow-inner focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
        />
        <span className={`absolute bottom-3 right-3 text-xs ${charCountColor}`}>
          {charCount}/{MAX_CHARS}
        </span>
      </div>

      {charCount > 0 && isUnderMin && (
        <p className="text-xs text-yellow-400">
          {MIN_CHARS - charCount} more character{MIN_CHARS - charCount !== 1 ? "s" : ""} needed
        </p>
      )}

      <button
        type="submit"
        disabled={isDisabled}
        className="flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white shadow transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? (
          <>
            <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Starting analysis...
          </>
        ) : (
          "Analyze Moonshot →"
        )}
      </button>
    </form>
  );
}
