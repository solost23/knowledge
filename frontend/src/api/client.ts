import type {
  ApiResponse,
  AskResult,
  Conversation,
  ConversationSummary,
  DocumentUploadJob,
  KnowledgeDocument,
} from '../types/api'

const API_BASE = '/api'

export class ApiError extends Error {
  constructor(message: string, readonly data: unknown) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const response = await fetch(`${API_BASE}${path}`, init)
  let body: ApiResponse<T> | undefined
  try {
    body = await response.json() as ApiResponse<T>
  } catch {
    // Proxies can return HTML error pages, such as Nginx's 413 response.
  }

  if (!response.ok || !body?.success) {
    const message = body?.message || (response.status === 413
      ? '文件过大，请将文件控制在 100 MiB 以内'
      : response.status === 504
        ? '服务处理超时，请刷新文档列表确认导入状态'
      : `请求失败 (${response.status})`)
    throw new ApiError(message, body?.data)
  }
  return body
}

function uploadFile(
  file: File,
  onProgress: (progress: number) => void,
): Promise<ApiResponse<DocumentUploadJob>> {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/upload`)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    }
    xhr.onload = () => {
      let body: ApiResponse<DocumentUploadJob> | undefined
      try {
        body = JSON.parse(xhr.responseText) as ApiResponse<DocumentUploadJob>
      } catch {
        // Reverse proxies may return an HTML error page.
      }

      if (xhr.status >= 200 && xhr.status < 300 && body?.success) {
        resolve(body)
        return
      }

      const message = body?.message || (xhr.status === 413
        ? '文件过大，请将文件控制在 100 MiB 以内'
        : xhr.status === 504
          ? '服务处理超时，请刷新文档列表确认导入状态'
          : `请求失败 (${xhr.status || '网络错误'})`)
      reject(new ApiError(message, body?.data))
    }
    xhr.onerror = () => reject(new ApiError('网络连接失败，无法上传文件', null))
    xhr.onabort = () => reject(new ApiError('文件上传已取消', null))
    xhr.send(form)
  })
}

export const api = {
  documents: () => request<KnowledgeDocument[]>('/docs'),
  conversations: () => request<ConversationSummary[]>('/conversations'),
  conversation: (id: string) => request<Conversation>(`/conversations/${encodeURIComponent(id)}`),
  createConversation: () => request<ConversationSummary>('/conversations', { method: 'POST' }),
  deleteConversation: (id: string) => request<null>(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  upload: uploadFile,
  uploadTask: (id: string) => request<DocumentUploadJob>(`/upload/tasks/${encodeURIComponent(id)}`),
  removeDocument: (name: string) => request<null>(`/docs/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  ask: (task: string, conversationId: string) => request<AskResult>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, conversation_id: conversationId }),
  }),
  resume: (executionId: string) => request<AskResult>('/agent/resume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ execution_id: executionId }),
  }),
  agentRun: (executionId: string) => request<AskResult>(`/agent/runs/${encodeURIComponent(executionId)}`),
  cancelRun: (executionId: string) => request<{ execution_id: string; status: string; cancel_requested: boolean }>(
    `/agent/runs/${encodeURIComponent(executionId)}/cancel`,
    { method: 'POST' },
  ),
  health: () => fetch(`${API_BASE}/healthz`).then((response) => response.ok),
}
