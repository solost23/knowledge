import type { AskResult, Conversation, ConversationSummary, DocumentUploadJob, KnowledgeDocument } from '../types/api'
import { apiBase, request, uploadFile } from './http'

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
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, conversation_id: conversationId }),
  }),
  resume: (executionId: string) => request<AskResult>('/agent/resume', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ execution_id: executionId }),
  }),
  agentRun: (executionId: string) => request<AskResult>(`/agent/runs/${encodeURIComponent(executionId)}`),
  cancelRun: (executionId: string) => request<{ execution_id: string; status: string; cancel_requested: boolean }>(
    `/agent/runs/${encodeURIComponent(executionId)}/cancel`, { method: 'POST' },
  ),
  health: () => fetch(`${apiBase}/healthz`).then((response) => response.ok),
}
