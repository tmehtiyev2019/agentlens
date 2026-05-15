const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type NodeStatus = "pending" | "running" | "complete" | "failed" | "hitl_pause";
export type PipelineStatus = "running" | "hitl_pause" | "complete" | "rejected" | "error";

export interface PipelineEvent {
  type:
    | "node_started"
    | "node_completed"
    | "node_failed"
    | "hitl_pause"
    | "pipeline_complete"
    | "pipeline_rejected";
  node: string | null;
  output: Record<string, unknown> | null;
  reason: string | null;
  thread_id: string;
}

export interface EvaluateResponse {
  thread_id: string;
  status: string;
}

export interface ResumeRequest {
  decision: "approved" | "rejected";
  comment?: string;
}

export interface ResumeResponse {
  thread_id: string;
  status: string;
}

export interface StatusResponse {
  thread_id: string;
  status: PipelineStatus;
  current_node: string | null;
  completed_nodes: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`AgentLens API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

export const api = {
  evaluate: (idea: string): Promise<EvaluateResponse> =>
    request<EvaluateResponse>("/evaluate", {
      method: "POST",
      body: JSON.stringify({ idea }),
    }),

  resume: (threadId: string, body: ResumeRequest): Promise<ResumeResponse> =>
    request<ResumeResponse>(`/evaluate/${encodeURIComponent(threadId)}/resume`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  status: (threadId: string): Promise<StatusResponse> =>
    request<StatusResponse>(`/evaluate/${encodeURIComponent(threadId)}/status`),

  report: (threadId: string): Promise<Record<string, unknown>> =>
    request<Record<string, unknown>>(`/evaluate/${encodeURIComponent(threadId)}/report`),

  streamUrl: (threadId: string): string =>
    `${API_URL}/evaluate/${encodeURIComponent(threadId)}/stream`,
};
