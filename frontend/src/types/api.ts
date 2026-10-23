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
  execution_id?: string
  status?: string
  mode: 'rag' | 'agent'
  route_reason?: string
  plan?: string[]
  current_step?: number
  steps?: AgentStep[]
  answer: string
  sources: SourceChunk[]
}
