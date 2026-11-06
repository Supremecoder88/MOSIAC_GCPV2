async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`/api/backend/${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    cache: "no-store",
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail || `API request failed: ${response.status}`
    );
  }

  return data as T;
}

export const api = {
  health: () => request<any>("health"),

  analyze: (payload: {
    task: string;
    nct_ids?: string[];
    max_studies?: number;
  }) =>
    request<any>("analyze", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  signals: (filters = "") =>
    request<any>(`signals${filters ? `?${filters}` : ""}`),

  signal: (id: string) =>
    request<any>(`signals/${encodeURIComponent(id)}`),

  reviewQueue: () => request<any>("review/queue"),

  review: (
    queueId: string,
    payload: {
      decision: "approve" | "reject" | "edit";
      reviewer: string;
      rejection_reason?: string;
      edit_summary?: string;
    }
  ) =>
    request<any>(`review/${encodeURIComponent(queueId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  episodes: (query = "") =>
    request<any>(`memory/episodes${query ? `?${query}` : ""}`),

  procedures: (agentName: string) =>
    request<any>(`memory/procedures/${encodeURIComponent(agentName)}`),

  sponsors: () => request<any>("sponsors"),

  sponsor: (name: string) =>
    request<any>(`sponsors/${encodeURIComponent(name)}`),
};