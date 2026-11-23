import type { ApiResponse, DocumentUploadJob } from '../types/api'

const API_BASE = '/api'

export class ApiError extends Error {
  constructor(message: string, readonly data: unknown) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const response = await fetch(`${API_BASE}${path}`, init)
  let body: ApiResponse<T> | undefined
  try { body = await response.json() as ApiResponse<T> } catch { /* proxy HTML */ }
  if (!response.ok || !body?.success) {
    const message = body?.message || (response.status === 413
      ? '文件超过服务端上传大小限制'
      : response.status === 504
        ? '服务处理超时，请刷新文档列表确认导入状态'
        : `请求失败 (${response.status})`)
    throw new ApiError(message, body?.data)
  }
  return body
}

export function uploadFile(file: File, onProgress: (progress: number) => void) {
  return new Promise<ApiResponse<DocumentUploadJob>>((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/upload`)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    xhr.onload = () => {
      let body: ApiResponse<DocumentUploadJob> | undefined
      try { body = JSON.parse(xhr.responseText) as ApiResponse<DocumentUploadJob> } catch { /* proxy HTML */ }
      if (xhr.status >= 200 && xhr.status < 300 && body?.success) return resolve(body)
      const message = body?.message || (xhr.status === 413
        ? '文件超过服务端上传大小限制'
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

export const apiBase = API_BASE
