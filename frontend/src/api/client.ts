import type { ApiResponse, AskResult, KnowledgeDocument } from '../types/api'

const API_BASE = '/api'

export class ApiError extends Error {
  constructor(message: string, readonly data: unknown) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const body = await response.json() as ApiResponse<T>
  if (!response.ok || !body.success) throw new ApiError(body.message || '请求失败', body.data)
  return body
}

export const api = {
  documents: () => request<KnowledgeDocument[]>('/docs'),
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<null>('/upload', { method: 'POST', body: form })
  },
  removeDocument: (name: string) => request<null>(`/docs/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  ask: (task: string) => request<AskResult>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task }),
  }),
  resume: (executionId: string) => request<AskResult>('/agent/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ execution_id: executionId }),
  }),
  health: () => fetch(`${API_BASE}/healthz`).then((response) => response.ok),
}
