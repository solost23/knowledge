import type { ApiResponse, DocumentUploadJob } from '../types/api'

const API_BASE = '/api'
const REQUEST_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  constructor(message: string, readonly data: unknown, readonly status = 0) {
    super(message)
    this.name = 'ApiError'
  }
}

async function withTimeout<T>(
  signal: AbortSignal | null | undefined,
  operation: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
): Promise<T> {
  const controller = new AbortController()
  const cancel = () => controller.abort(signal?.reason)
  if (signal?.aborted) cancel()
  else signal?.addEventListener('abort', cancel, { once: true })
  const timer = setTimeout(() => controller.abort(new DOMException('请求超时，请重新连接查看状态', 'TimeoutError')), timeoutMs)
  try {
    controller.signal.throwIfAborted()
    return await operation(controller.signal)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', cancel)
  }
}

export async function request<T>(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<ApiResponse<T>> {
  return withTimeout(init?.signal, async (signal) => {
    const response = await fetch(`${API_BASE}${path}`, { ...init, signal })
    let body: ApiResponse<T> | undefined
    try { body = await response.json() as ApiResponse<T> } catch {
      if (signal.aborted) throw signal.reason
      // 代理可能返回 HTML 而不是 JSON。
    }
    if (!response.ok || !body?.success) {
      const message = body?.message || (response.status === 413
        ? '文件超过服务端上传大小限制'
        : response.status === 504
          ? '服务处理超时，请重新连接确认任务状态'
          : `请求失败 (${response.status})`)
      throw new ApiError(message, body?.data, response.status)
    }
    return body
  }, timeoutMs)
}

export function healthCheck(signal?: AbortSignal) {
  return withTimeout(signal, async (requestSignal) => {
    const response = await fetch(`${API_BASE}/healthz`, { signal: requestSignal })
    await response.body?.cancel()
    return response.ok
  }, REQUEST_TIMEOUT_MS)
}

export function uploadFile(file: File, onProgress: (progress: number) => void, signal?: AbortSignal) {
  return new Promise<ApiResponse<DocumentUploadJob>>((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason)
    const form = new FormData()
    form.append('file', file)
    const xhr = new XMLHttpRequest()
    const cancel = () => xhr.abort()
    const cleanup = () => signal?.removeEventListener('abort', cancel)
    const fail = (cause: unknown) => { cleanup(); reject(cause) }
    xhr.open('POST', `${API_BASE}/upload`)
    xhr.timeout = 120_000
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    xhr.onload = () => {
      cleanup()
      let body: ApiResponse<DocumentUploadJob> | undefined
      try { body = JSON.parse(xhr.responseText) as ApiResponse<DocumentUploadJob> } catch { /* 代理可能返回 HTML。 */ }
      if (xhr.status >= 200 && xhr.status < 300 && body?.success) return resolve(body)
      const message = body?.message || (xhr.status === 413
        ? '文件超过服务端上传大小限制'
        : xhr.status === 504
          ? '服务处理超时，请刷新文档列表确认导入状态'
          : `请求失败 (${xhr.status || '网络错误'})`)
      fail(new ApiError(message, body?.data, xhr.status))
    }
    xhr.onerror = () => fail(new ApiError('网络连接失败，无法上传文件', null))
    xhr.onabort = () => fail(signal?.reason ?? new DOMException('文件上传已取消', 'AbortError'))
    xhr.ontimeout = () => fail(new ApiError('上传超时，请刷新文档列表确认导入状态', null))
    signal?.addEventListener('abort', cancel, { once: true })
    try { xhr.send(form) } catch (cause) { fail(cause) }
  })
}
