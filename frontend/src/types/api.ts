export interface ApiResponse<T> {
  code: number
  success: boolean
  message: string
  data: T
}

export interface KnowledgeDocument {
  id: number
  name: string
  created_at: string
}

export interface DocumentUploadJob {
  id: string
  filename: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  stage: string
  error: string | null
  created_at: string
  updated_at: string
}

export interface ConversationSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  result: AskResult | null
  created_at: string
}

export interface Conversation extends ConversationSummary {
  messages: ConversationMessage[]
}

export interface SourceChunk {
  source: string
  content: string
}

export interface AgentStep {
  index: number
  task: string
  result: string
  tools: string[]
}

export interface AskResult {
  task?: string
  conversation_id?: string
  execution_id?: string
  status?: 'running' | 'completed' | 'failed' | 'cancelled' | string
  mode: 'routing' | 'rag' | 'agent'
  route_reason?: string
  current_action?: string
  active_step?: number | null
  resumable?: boolean
  cancel_requested?: boolean
  started_at?: string
  finished_at?: string | null
  duration_ms?: number
  result?: { error_message?: string }
  plan?: string[]
  current_step?: number
  steps?: AgentStep[]
  answer: string
  sources: SourceChunk[]
}
