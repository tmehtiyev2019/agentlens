"use client";

import { useState } from "react";
import { api, type ResumeResponse } from "@/lib/api";

export interface HITLResumeResult {
  resume: (decision: "approved" | "rejected", comment?: string) => Promise<ResumeResponse | null>;
  isSubmitting: boolean;
  error: string | null;
}

export function useHITLResume(threadId: string): HITLResumeResult {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resume = async (
    decision: "approved" | "rejected",
    comment?: string,
  ): Promise<ResumeResponse | null> => {
    if (isSubmitting) return null;

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await api.resume(threadId, { decision, comment });
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to resume pipeline";
      setError(message);
      return null;
    } finally {
      setIsSubmitting(false);
    }
  };

  return { resume, isSubmitting, error };
}
