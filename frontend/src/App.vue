<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'
import {
  Activity, ArrowDown, ArrowUp, BookOpen, Check, CheckCircle2, Copy,
  FileText, FolderOpen, LoaderCircle, MessageSquarePlus,
  PanelLeft, PanelRightClose, Plus, RefreshCw, Search, ShieldCheck, Sparkles, StopCircle, Trash2,
  Upload, X,
} from 'lucide-vue-next'
import { api, ApiError } from './api/client'
import type {
  AskResult,
  ConversationSummary,
  DocumentUploadJob,
  KnowledgeDocument,
  SourceChunk,
} from './types/api'

interface Message {
  id: number | string
  role: 'user' | 'assistant'
  text: string
  result?: AskResult
}

interface ImportProgress {
  filename: string
  status: 'uploading' | DocumentUploadJob['status']
  progress: number
  uploadProgress: number
  stage: string
  error?: string
}

const markdown = new MarkdownIt({ breaks: true, linkify: true })
const uploadTaskStorageKey = 'knowledge.activeDocumentUpload'
const documents = ref<KnowledgeDocument[]>([])
const conversations = ref<ConversationSummary[]>([])
const messages = ref<Message[]>([])
const task = ref('')
const loading = ref(false)
const uploading = ref(false)
const importProgress = ref<ImportProgress | null>(null)
const connected = ref(false)
const error = ref('')
const failedExecutionId = ref('')
const activeExecutionId = ref('')
const stopRequested = ref(false)
const copiedMessageId = ref<number | string | null>(null)
const elapsedNow = ref(Date.now())
let elapsedTimer: number | undefined
const filter = ref('')
const selectedSource = ref<SourceChunk | null>(null)
const showConversationDrawer = ref(false)
const showDetails = ref(true)
const showMobileDetails = ref(false)
const isMobile = ref(window.matchMedia('(max-width: 620px)').matches)
const composer = ref<HTMLTextAreaElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const conversationScroll = ref<HTMLElement | null>(null)
const activeConversationId = ref<string | null>(null)
const detailsOpen = computed(() => isMobile.value ? showMobileDetails.value : showDetails.value)
const visibleDocuments = computed(() => {
  const query = filter.value.trim().toLocaleLowerCase()
  return query ? documents.value.filter((doc) => doc.name.toLocaleLowerCase().includes(query)) : documents.value
})
const latestRun = computed(() => [...messages.value].reverse().find((message) => message.result?.mode === 'agent')?.result)
const activeExecution = computed(() => [...messages.value].reverse().find((message) => (
  message.result?.status === 'running'
))?.result)

function renderMarkdown(content: string) {
  return markdown.render(content)
}

function formatDuration(result?: AskResult) {
  if (!result?.started_at && result?.duration_ms === undefined) return ''
  const elapsed = result.status === 'running' && result.started_at
    ? Math.max(0, elapsedNow.value - new Date(result.started_at).getTime())
    : result.duration_ms ?? 0
  if (elapsed < 60_000) return `${(elapsed / 1000).toFixed(1)} 秒`
  const minutes = Math.floor(elapsed / 60_000)
  const seconds = Math.floor((elapsed % 60_000) / 1000)
  return `${minutes} 分 ${seconds} 秒`
}

async function copyQuestion(message: Message) {
  try {
    await navigator.clipboard.writeText(message.text)
    copiedMessageId.value = message.id
    window.setTimeout(() => {
      if (copiedMessageId.value === message.id) copiedMessageId.value = null
    }, 1400)
  } catch {
    error.value = '复制失败，请检查浏览器剪贴板权限'
  }
}

function scrollConversation(force = false) {
  const element = conversationScroll.value
  if (!element) return
  const nearBottom = element.scrollHeight - element.clientHeight - element.scrollTop < 160
  if (force || nearBottom) element.scrollTop = element.scrollHeight
}

watch(messages, () => scrollConversation(), { deep: true, flush: 'post' })

async function refreshDocuments() {
  try {
    documents.value = (await api.documents()).data
    connected.value = true
  } catch (cause) {
    connected.value = false
    error.value = cause instanceof Error ? cause.message : '无法连接服务'
  }
}

async function checkConnection() {
  try { connected.value = await api.health() } catch { connected.value = false }
}

async function refreshConversations() {
  try {
    conversations.value = (await api.conversations()).data
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '无法读取对话列表'
  }
}

async function selectConversation(id: string) {
  if (loading.value || id === activeConversationId.value) return
  error.value = ''
  try {
    const conversation = (await api.conversation(id)).data
    activeConversationId.value = conversation.id
    messages.value = conversation.messages.map((message) => ({
      id: message.id,
      role: message.role,
      text: message.content,
      result: message.result ?? undefined,
    }))
    failedExecutionId.value = ''
    selectedSource.value = null
    showConversationDrawer.value = false
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '无法读取对话'
  }
}

async function sendTask() {
  const prompt = task.value.trim()
  if (!prompt || loading.value) return
  error.value = ''
  task.value = ''
  const userMessage: Message = { id: Date.now(), role: 'user', text: prompt }
  messages.value.push(userMessage)
  loading.value = true
  await nextTick()
  scrollConversation(true)
  try {
    let conversationId = activeConversationId.value
    if (!conversationId) {
      const created = (await api.createConversation()).data
      conversationId = created.id
      activeConversationId.value = conversationId
      conversations.value.unshift(created)
    }
    const result = (await api.ask(prompt, conversationId)).data
    failedExecutionId.value = ''
    activeExecutionId.value = result.execution_id ?? ''
    stopRequested.value = false
    const assistantMessage: Message = {
      id: result.execution_id ?? Date.now() + 1,
      role: 'assistant',
      text: result.answer,
      result,
    }
    messages.value.push(assistantMessage)
    if (result.status === 'running' && result.execution_id) {
      await trackRun(assistantMessage)
    } else {
      await refreshConversations()
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '任务执行失败'
    failedExecutionId.value = cause instanceof ApiError
      && typeof cause.data === 'object'
      && cause.data !== null
      && 'execution_id' in cause.data
      ? String(cause.data.execution_id)
      : ''
  } finally {
    activeExecutionId.value = ''
    stopRequested.value = false
    loading.value = false
    await nextTick()
    composer.value?.focus()
  }
}

async function trackRun(message: Message) {
  const run = message.result
  if (!run?.execution_id) return
  let connectionFailures = 0
  while (run.status === 'running') {
    await new Promise((resolve) => window.setTimeout(resolve, 1000))
    try {
      const progress = (await api.agentRun(run.execution_id)).data
      connectionFailures = 0
      Object.assign(run, progress, {
        mode: progress.mode ?? run.mode,
        route_reason: progress.route_reason ?? run.route_reason,
        conversation_id: progress.conversation_id ?? run.conversation_id,
      })
      if (progress.status === 'completed') {
        message.text = progress.answer
        failedExecutionId.value = ''
        await refreshConversations()
      } else if (progress.status === 'failed') {
        message.text = progress.result?.error_message || '任务执行中断。'
        failedExecutionId.value = progress.resumable ? run.execution_id : ''
        error.value = progress.resumable
          ? 'Agent 执行中断，可以恢复任务。'
          : message.text
      } else if (progress.status === 'cancelled') {
        message.text = '任务已停止'
        failedExecutionId.value = ''
        await refreshConversations()
      }
      await nextTick()
      scrollConversation()
    } catch {
      connectionFailures += 1
      run.current_action = connectionFailures >= 5
        ? '状态连接中断，正在重试…'
        : '正在读取 Agent 执行状态…'
    }
  }
}

async function stopTask() {
  if (!activeExecutionId.value || stopRequested.value) return
  stopRequested.value = true
  if (activeExecution.value) {
    activeExecution.value.cancel_requested = true
    activeExecution.value.current_action = '正在停止，等待当前模型或工具调用返回'
  }
  try {
    await api.cancelRun(activeExecutionId.value)
  } catch (cause) {
    stopRequested.value = false
    error.value = cause instanceof Error ? cause.message : '停止任务失败'
  }
}

async function resumeTask() {
  if (!failedExecutionId.value || loading.value) return
  loading.value = true
  error.value = ''
  try {
    const result = (await api.resume(failedExecutionId.value)).data
    failedExecutionId.value = ''
    const agentResult = { ...result, mode: 'agent' as const }
    messages.value.push({ id: Date.now(), role: 'assistant', text: result.answer, result: agentResult })
    await refreshConversations()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '恢复任务失败'
  } finally {
    loading.value = false
  }
}

function onComposerKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void sendTask()
  }
}

async function uploadFiles(event: Event) {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files ?? [])
  if (!files.length) return
  uploading.value = true
  error.value = ''
  for (const file of files) {
    importProgress.value = {
      filename: file.name,
      status: 'uploading',
      progress: 0,
      uploadProgress: 0,
      stage: '正在上传文件',
    }
    try {
      const accepted = (await api.upload(file, (progress) => {
        if (importProgress.value?.filename === file.name) {
          importProgress.value.uploadProgress = progress
        }
      })).data
      localStorage.setItem(uploadTaskStorageKey, accepted.id)
      const finished = await trackDocumentImport(accepted)
      if (!finished) break
      if (finished.status === 'failed') throw new Error(finished.error || '文档导入失败')
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : `导入 ${file.name} 失败`
      error.value = message
      if (importProgress.value) {
        importProgress.value.status = 'failed'
        importProgress.value.stage = '导入失败'
        importProgress.value.error = message
      }
      break
    }
  }
  await refreshDocuments()
  uploading.value = false
  input.value = ''
}

function applyImportJob(job: DocumentUploadJob) {
  importProgress.value = {
    filename: job.filename,
    status: job.status,
    progress: job.progress,
    uploadProgress: 100,
    stage: job.stage,
    error: job.error ?? undefined,
  }
}

async function trackDocumentImport(initialJob: DocumentUploadJob): Promise<DocumentUploadJob | null> {
  let job = initialJob
  let connectionFailures = 0
  applyImportJob(job)

  while (job.status === 'queued' || job.status === 'processing') {
    await new Promise((resolve) => window.setTimeout(resolve, 1000))
    try {
      job = (await api.uploadTask(job.id)).data
      connectionFailures = 0
      applyImportJob(job)
    } catch {
      connectionFailures += 1
      if (importProgress.value) importProgress.value.stage = '连接暂时中断，正在重试获取进度'
      if (connectionFailures >= 5) {
        if (importProgress.value) {
          importProgress.value.stage = '暂时无法获取状态，刷新页面可继续查看'
        }
        return null
      }
    }
  }

  localStorage.removeItem(uploadTaskStorageKey)
  if (job.status === 'completed') await refreshDocuments()
  return job
}

async function resumeDocumentImport() {
  const jobId = localStorage.getItem(uploadTaskStorageKey)
  if (!jobId) return
  try {
    const job = (await api.uploadTask(jobId)).data
    if (job.status === 'queued' || job.status === 'processing') uploading.value = true
    const finished = await trackDocumentImport(job)
    if (finished?.status === 'failed') error.value = finished.error || '文档导入失败'
  } catch {
    localStorage.removeItem(uploadTaskStorageKey)
  } finally {
    uploading.value = false
  }
}

async function removeDocument(doc: KnowledgeDocument) {
  if (!window.confirm(`从知识库删除“${doc.name}”？`)) return
  try {
    await api.removeDocument(doc.name)
    await refreshDocuments()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '删除失败'
  }
}

function newConversation() {
  if (loading.value) return
  activeConversationId.value = null
  messages.value = []
  error.value = ''
  failedExecutionId.value = ''
  showConversationDrawer.value = false
  task.value = ''
  composer.value?.focus()
}

async function removeConversation(conversation: ConversationSummary) {
  if (!window.confirm(`删除对话“${conversation.title}”？`)) return
  try {
    await api.deleteConversation(conversation.id)
    conversations.value = conversations.value.filter((item) => item.id !== conversation.id)
    if (activeConversationId.value === conversation.id) newConversation()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '删除对话失败'
  }
}

function toggleDetails() {
  if (isMobile.value) showMobileDetails.value = !showMobileDetails.value
  else showDetails.value = !showDetails.value
}

function inspectSource(source: SourceChunk) {
  selectedSource.value = source
  showDetails.value = true
  showMobileDetails.value = true
}

const mobileQuery = window.matchMedia('(max-width: 620px)')
function updateViewport(event: MediaQueryListEvent) {
  isMobile.value = event.matches
}

function formatDate(value: string) {
  const date = new Date(value.replace(' ', 'T'))
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric' }).format(date)
}

onMounted(() => {
  elapsedTimer = window.setInterval(() => { elapsedNow.value = Date.now() }, 1000)
  mobileQuery.addEventListener('change', updateViewport)
  void refreshDocuments()
  void checkConnection()
  void resumeDocumentImport()
  void (async () => {
    await refreshConversations()
    if (conversations.value.length) await selectConversation(conversations.value[0]!.id)
  })()
})

onBeforeUnmount(() => {
  mobileQuery.removeEventListener('change', updateViewport)
  if (elapsedTimer !== undefined) window.clearInterval(elapsedTimer)
})
</script>

<template>
  <div class="app-shell">
    <div v-if="showConversationDrawer" class="sidebar-scrim" @click="showConversationDrawer = false"></div>
    <aside :class="['sidebar', { 'sidebar-open': showConversationDrawer }]">
      <div class="brand-row">
        <div class="brand-mark"><BookOpen :size="17" :stroke-width="2.2" /></div>
        <span class="brand-name">知序</span>
        <button class="icon-button sidebar-close" title="关闭会话列表" aria-label="关闭会话列表" @click="showConversationDrawer = false"><X :size="16" /></button>
      </div>

      <button class="new-chat" :disabled="loading" @click="newConversation"><MessageSquarePlus :size="16" />新建对话 <kbd>⌘ K</kbd></button>

      <div class="nav-label">工作区</div>
      <nav class="primary-nav" aria-label="工作区导航">
        <button class="nav-item active"><FolderOpen :size="16" /><span>知识库</span><span class="nav-count">{{ documents.length }}</span></button>
        <button class="nav-item" :disabled="loading" @click="newConversation"><MessageSquarePlus :size="16" /><span>新建对话</span></button>
      </nav>

      <div class="history-heading"><span>最近对话</span></div>
      <div class="history-list">
        <div v-for="conversation in conversations" :key="conversation.id" :class="['history-row', { selected: conversation.id === activeConversationId }]">
          <button class="history-item" :title="conversation.title" :disabled="loading" @click="selectConversation(conversation.id)"><span class="history-dot"></span><span class="truncate">{{ conversation.title }}</span></button>
          <button class="history-delete" :title="`删除对话 ${conversation.title}`" :aria-label="`删除对话 ${conversation.title}`" :disabled="loading" @click="removeConversation(conversation)"><Trash2 :size="13" /></button>
        </div>
        <div v-if="conversations.length === 0" class="empty-history">还没有对话</div>
      </div>

      <div class="sidebar-bottom">
        <div class="local-status"><span :class="['status-dot', { online: connected }]"></span><span>{{ connected ? '本地服务已连接' : '服务未连接' }}</span><ShieldCheck :size="14" /></div>
        <div class="profile-row"><div class="avatar">知</div><span class="profile-name">本地工作区</span></div>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
        <button class="icon-button conversation-menu-toggle" title="查看对话" aria-label="查看对话" @click="showConversationDrawer = true"><PanelLeft :size="17" /></button>
        <div class="breadcrumb"><span>工作区</span><span class="crumb-divider">/</span><strong>知识库</strong></div>
        <div class="topbar-actions">
          <div class="index-state"><span :class="['status-dot', { online: connected }]"></span><span>本机索引</span><span class="state-separator">·</span><strong>{{ documents.length }} 份文档</strong></div>
          <button class="icon-button details-toggle" :title="detailsOpen ? '收起详情' : '显示详情'" :aria-label="detailsOpen ? '收起详情' : '显示详情'" @click="toggleDetails"><PanelRightClose :size="17" /></button>
        </div>
      </header>

      <section class="workspace">
        <div class="conversation-column">
          <div v-if="messages.length === 0" class="welcome-area">
            <div class="welcome-icon"><Sparkles :size="19" /></div>
            <p class="eyebrow">私人知识工作台</p>
            <h1>把资料理清楚，<span>再开始行动。</span></h1>
            <p class="welcome-copy">向你的知识库提问。简单问题直接检索，多文档任务会自动规划并逐步执行。</p>
            <div class="suggestions">
              <button @click="task = '总结最近上传的文档，并按主题归类'">总结最近上传的文档 <ArrowDown :size="14" /></button>
              <button @click="task = '比较两个版本的报销制度，整理差异和来源'">比较制度版本 <ArrowDown :size="14" /></button>
              <button @click="task = '知识库里有哪些关于发布流程的内容？'">查找发布流程 <ArrowDown :size="14" /></button>
            </div>
          </div>

          <div v-else ref="conversationScroll" class="conversation" :key="activeConversationId ?? 'new-conversation'">
            <article v-for="message in messages" :key="message.id" :class="['message-row', message.role]">
              <div v-if="message.role === 'assistant'" class="assistant-avatar"><Sparkles :size="15" /></div>
              <div class="message-content">
                <div v-if="message.role === 'user'" class="user-message-wrap">
                  <div class="user-message">{{ message.text }}</div>
                  <button
                    class="icon-button user-copy-button"
                    :title="copiedMessageId === message.id ? '已复制' : '复制提问'"
                    :aria-label="copiedMessageId === message.id ? '已复制提问' : '复制提问'"
                    @click="copyQuestion(message)"
                  ><Check v-if="copiedMessageId === message.id" :size="14" /><Copy v-else :size="14" /></button>
                </div>
                <template v-else>
                  <div class="answer-meta">
                    <span class="answer-brand">知序</span>
                    <span v-if="message.result?.mode === 'agent'" class="mode-label"><Activity :size="12" />任务执行</span>
                    <span v-else-if="message.result?.mode === 'routing'" class="mode-label rag-label"><LoaderCircle class="spin" :size="12" />判断中</span>
                    <span v-else class="mode-label rag-label"><Search :size="12" />知识检索</span>
                    <span v-if="message.result?.started_at || message.result?.duration_ms !== undefined" class="duration-label">耗时 {{ formatDuration(message.result) }}</span>
                  </div>
                  <div v-if="message.result?.route_reason" class="route-reason">{{ message.result.route_reason }}</div>
                  <div v-if="message.text" class="answer-text markdown-body" v-html="renderMarkdown(message.text)"></div>
                  <div v-if="message.result?.status === 'running'" class="agent-live-status">
                    <LoaderCircle class="spin" :size="14" />
                    <span>{{ message.result.current_action || '正在处理请求…' }}</span>
                  </div>
                  <div v-if="message.result?.status === 'cancelled'" class="cancelled-status">任务已停止</div>
                  <div v-if="message.result?.mode === 'agent' && message.result.plan?.length" class="plan-block">
                    <div class="block-heading"><span>执行计划</span><span class="step-progress">{{ message.result.steps?.length ?? 0 }}/{{ message.result.plan.length }} 步完成</span></div>
                    <div v-for="(step, index) in message.result.plan" :key="step" class="plan-step">
                      <div :class="['step-check', { done: index < (message.result.current_step ?? message.result.steps?.length ?? 0), active: message.result.active_step === index + 1 }]"><Check v-if="index < (message.result.current_step ?? message.result.steps?.length ?? 0)" :size="12" /><LoaderCircle v-else-if="message.result.active_step === index + 1" class="spin" :size="11" /></div>
                      <div class="step-detail"><span class="step-title">{{ step }}</span><span v-if="message.result.steps?.[index]" class="step-result markdown-body" v-html="renderMarkdown(message.result.steps[index]?.result ?? '')"></span></div>
                    </div>
                  </div>
                  <div v-if="message.result?.sources?.length" class="sources-block">
                    <div class="block-heading"><span>参考来源</span><span class="source-count">{{ message.result.sources.length }}</span></div>
                    <button v-for="(source, index) in message.result.sources" :key="`${source.source}-${index}`" class="source-row" @click="inspectSource(source)">
                      <FileText :size="14" /><span class="truncate">{{ source.source }}</span><span class="source-index">{{ String(index + 1).padStart(2, '0') }}</span>
                    </button>
                  </div>
                </template>
              </div>
            </article>
            <div v-if="loading && !activeExecution" class="loading-row"><div class="assistant-avatar"><Sparkles :size="15" /></div><LoaderCircle class="spin" :size="17" /><span>正在创建执行任务…</span></div>
          </div>

          <div class="composer-area">
            <div v-if="importProgress" :class="['import-progress-panel', `import-${importProgress.status}`]">
              <div class="import-progress-heading">
                <LoaderCircle v-if="['uploading', 'queued', 'processing'].includes(importProgress.status)" class="spin" :size="14" />
                <CheckCircle2 v-else-if="importProgress.status === 'completed'" :size="14" />
                <X v-else :size="14" />
                <span class="import-filename" :title="importProgress.filename">{{ importProgress.filename }}</span>
                <strong>{{ importProgress.status === 'uploading' ? importProgress.uploadProgress : importProgress.progress }}%</strong>
                <button v-if="!uploading" class="icon-button" title="关闭导入状态" aria-label="关闭导入状态" @click="importProgress = null"><X :size="14" /></button>
              </div>
              <div
                class="import-progress-track"
                role="progressbar"
                :aria-valuenow="importProgress.status === 'uploading' ? importProgress.uploadProgress : importProgress.progress"
                aria-valuemin="0"
                aria-valuemax="100"
              >
                <div
                  class="import-progress-fill"
                  :style="{ width: `${importProgress.status === 'uploading' ? importProgress.uploadProgress : importProgress.progress}%` }"
                ></div>
              </div>
              <div class="import-progress-stage">
                <span>{{ importProgress.stage }}</span>
                <span v-if="importProgress.error" class="import-progress-error">{{ importProgress.error }}</span>
              </div>
            </div>
            <div v-if="error" class="error-banner"><span>{{ error }}</span><button v-if="failedExecutionId" class="resume-button" :disabled="loading" @click="resumeTask">恢复任务</button><button class="icon-button" aria-label="关闭错误" @click="error = ''; failedExecutionId = ''"><X :size="15" /></button></div>
            <div class="composer">
              <textarea ref="composer" v-model="task" rows="1" placeholder="向你的知识库提问，或描述一项需要完成的任务…" :disabled="loading" @keydown="onComposerKeydown"></textarea>
              <div class="composer-bottom">
                <div class="composer-tools">
                  <input ref="fileInput" type="file" accept=".pdf,.docx,.pptx,.xlsx,.md" multiple hidden @change="uploadFiles" />
                  <button class="tool-button" :disabled="uploading" title="上传文档" @click="fileInput?.click()"><Upload v-if="!uploading" :size="15" /><LoaderCircle v-else class="spin" :size="15" /><span>{{ uploading ? '正在导入' : '添加资料' }}</span></button>
                  <span class="tool-divider"></span><span class="composer-hint"><kbd>↵</kbd> 发送 <span>·</span> <kbd>⇧ ↵</kbd> 换行</span>
                </div>
                <button v-if="loading && activeExecutionId" class="stop-button" :disabled="stopRequested" :aria-label="stopRequested ? '正在停止' : '停止执行'" :title="stopRequested ? '正在停止…' : '停止执行'" @click="stopTask"><LoaderCircle v-if="stopRequested" class="spin" :size="16" /><StopCircle v-else :size="17" /></button>
                <button v-else class="send-button" :disabled="!task.trim() || loading" aria-label="发送" title="发送" @click="sendTask"><ArrowUp :size="17" /></button>
              </div>
            </div>
            <div class="privacy-note"><ShieldCheck :size="12" />文档与执行记录保存在本机</div>
          </div>
        </div>

        <aside v-if="detailsOpen" :class="['details-panel', { 'mobile-open': isMobile && showMobileDetails }]">
          <section v-if="selectedSource" class="source-inspector">
            <div class="panel-heading"><div><span class="panel-kicker">来源片段</span><h2>{{ selectedSource.source }}</h2></div><button class="icon-button" aria-label="关闭来源" @click="selectedSource = null"><X :size="16" /></button></div>
            <div class="source-preview">{{ selectedSource.content }}</div>
          </section>
          <template v-else>
            <section class="details-section knowledge-section">
              <div class="panel-heading"><div><span class="panel-kicker">资料空间</span><h2>知识库</h2></div><button class="icon-button" title="添加文档" aria-label="添加文档" @click="fileInput?.click()"><Plus :size="17" /></button></div>
              <div class="doc-summary"><div class="summary-icon"><BookOpen :size="16" /></div><div><strong>{{ documents.length }}</strong><span>份已索引文档</span></div><button class="icon-button refresh-button" title="刷新列表" aria-label="刷新列表" @click="refreshDocuments"><RefreshCw :size="14" /></button></div>
              <label class="doc-search"><Search :size="14" /><input v-model="filter" placeholder="筛选文档" /></label>
              <div class="document-list">
                <div v-for="doc in visibleDocuments" :key="doc.id" class="document-item">
                  <div class="document-icon"><FileText :size="15" /></div><div class="document-info"><span class="document-name" :title="doc.name">{{ doc.name }}</span><span class="document-date">{{ formatDate(doc.created_at) }}</span></div>
                  <button class="icon-button delete-button" :title="`删除 ${doc.name}`" :aria-label="`删除 ${doc.name}`" @click="removeDocument(doc)"><Trash2 :size="14" /></button>
                </div>
                <div v-if="visibleDocuments.length === 0" class="document-empty">{{ filter ? '没有匹配的文档' : '上传文档后，这里会显示索引' }}</div>
              </div>
              <button class="upload-drop" :disabled="uploading" @click="fileInput?.click()"><Upload :size="15" /><span>{{ uploading ? '正在导入…' : '导入文档' }}</span><span class="upload-types">PDF · DOCX · PPTX · XLSX · MD</span></button>
            </section>

            <section class="details-section run-section">
              <div class="panel-heading"><div><span class="panel-kicker">任务运行时</span><h2>最近执行</h2></div><span v-if="latestRun?.status === 'running'" class="run-status running"><LoaderCircle class="spin" :size="13" />执行中</span><span v-else-if="latestRun?.status === 'failed'" class="run-status failed"><X :size="13" />已中断</span><span v-else-if="latestRun" class="run-status complete"><CheckCircle2 :size="13" />已完成</span><span v-else class="run-status idle"><span class="status-dot"></span>待命</span></div>
              <template v-if="latestRun">
                <div class="run-task">{{ latestRun.answer || latestRun.task }}</div>
                <div v-if="latestRun.status === 'running' && latestRun.current_action" class="run-current-action">{{ latestRun.current_action }}</div>
                <div class="run-stats"><span>{{ latestRun.plan?.length ?? 0 }} 个步骤</span><span>{{ latestRun.sources?.length ?? 0 }} 个来源</span></div>
                <div v-for="(step, index) in latestRun.plan" :key="step" class="run-step"><span :class="['mini-check', { pending: index >= (latestRun.current_step ?? latestRun.steps?.length ?? 0) }]"><Check v-if="index < (latestRun.current_step ?? latestRun.steps?.length ?? 0)" :size="10" /><LoaderCircle v-else-if="latestRun.active_step === index + 1" class="spin" :size="10" /></span><span class="truncate">{{ step }}</span></div>
              </template>
              <div v-else class="run-empty">提交复杂任务后，执行计划与步骤结果会显示在这里。</div>
            </section>

            <div class="security-note"><ShieldCheck :size="15" /><div><strong>本地优先</strong><span>只有模型请求会发送任务文本；索引和执行记录保留在此设备。</span></div></div>
          </template>
        </aside>
      </section>
    </main>
  </div>
</template>
