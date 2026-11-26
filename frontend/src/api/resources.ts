import type { AskResult, Conversation, ConversationSummary, DocumentUploadJob, KnowledgeDocument } from '../types/api'
import { healthCheck, request, uploadFile } from './http'

export const api = {
  documents: (signal?: AbortSignal) => request<KnowledgeDocument[]>('/docs', { signal }),
  conversations: (signal?: AbortSignal) => request<ConversationSummary[]>('/conversations', { signal }),
  conversation: (id: string, signal?: AbortSignal) => request<Conversation>(`/conversations/${encodeURIComponent(id)}`, { signal }),
  createConversation: (signal?: AbortSignal) => request<ConversationSummary>('/conversations', { method: 'POST', signal }),
  deleteConversation: (id: string, signal?: AbortSignal) => request<null>(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE', signal }),
  upload: uploadFile,
  uploadTask: (id: string, signal?: AbortSignal) => request<DocumentUploadJob>(`/upload/tasks/${encodeURIComponent(id)}`, { signal }),
  removeDocument: (name: string, signal?: AbortSignal) => request<null>(`/docs/${encodeURIComponent(name)}`, { method: 'DELETE', signal }),
  ask: (task: string, conversationId: string, signal?: AbortSignal) => request<AskResult>('/ask', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
    body: JSON.stringify({ task, conversation_id: conversationId }),
  }),
  resume: (executionId: string, signal?: AbortSignal) => request<AskResult>('/agent/resume', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
    body: JSON.stringify({ execution_id: executionId }),
  }),
  agentRun: (executionId: string, signal?: AbortSignal) => request<AskResult>(`/agent/runs/${encodeURIComponent(executionId)}`, { signal }),
  cancelRun: (executionId: string, signal?: AbortSignal) => request<{ execution_id: string; status: string; cancel_requested: boolean }>(
    `/agent/runs/${encodeURIComponent(executionId)}/cancel`, { method: 'POST', signal },
  ),
  health: healthCheck,
}
