import { ApiError } from './http'

export function isPermanentError(cause: unknown) {
  return cause instanceof ApiError && cause.status >= 400 && cause.status < 500
    && ![408, 425, 429].includes(cause.status)
}

export function retryDelay(failures: number) {
  return Math.min(1000 * 2 ** failures, 8000)
}

export function waitForPoll(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason)
    const cancel = () => {
      clearTimeout(timer)
      reject(signal.reason)
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', cancel)
      resolve()
    }, milliseconds)
    signal.addEventListener('abort', cancel, { once: true })
  })
}

export function clearStoredTask(storage: Storage, key: string, taskId: string) {
  if (storage.getItem(key) === taskId) storage.removeItem(key)
}
