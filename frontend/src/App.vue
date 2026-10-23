<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  Activity, ArrowDown, ArrowUp, BookOpen, Check, CheckCircle2,
  FileText, FolderOpen, LoaderCircle, MessageSquarePlus,
  PanelRightClose, Plus, RefreshCw, Search, ShieldCheck, Sparkles, Trash2,
  Upload, X,
} from 'lucide-vue-next'
import { api, ApiError } from './api/client'
import type { AskResult, KnowledgeDocument, SourceChunk } from './types/api'

interface Message {
  id: number
  role: 'user' | 'assistant'
  text: string
  result?: AskResult
}

const documents = ref<KnowledgeDocument[]>([])
const messages = ref<Message[]>([])
const task = ref('')
const loading = ref(false)
const uploading = ref(false)
const connected = ref(false)
const error = ref('')
const failedExecutionId = ref('')
const filter = ref('')
const selectedSource = ref<SourceChunk | null>(null)
const showDetails = ref(true)
const showMobileDetails = ref(false)
const isMobile = ref(window.matchMedia('(max-width: 620px)').matches)
const composer = ref<HTMLTextAreaElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const threadId = ref(1)
const detailsOpen = computed(() => isMobile.value ? showMobileDetails.value : showDetails.value)
const visibleDocuments = computed(() => {
  const query = filter.value.trim().toLocaleLowerCase()
  return query ? documents.value.filter((doc) => doc.name.toLocaleLowerCase().includes(query)) : documents.value
})
const latestRun = computed(() => [...messages.value].reverse().find((message) => message.result?.mode === 'agent')?.result)

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

async function sendTask() {
  const prompt = task.value.trim()
  if (!prompt || loading.value) return
  error.value = ''
  task.value = ''
  const userMessage: Message = { id: Date.now(), role: 'user', text: prompt }
  messages.value.push(userMessage)
  loading.value = true
  await nextTick()
  try {
    const result = (await api.ask(prompt)).data
    failedExecutionId.value = ''
    messages.value.push({ id: Date.now() + 1, role: 'assistant', text: result.answer, result })
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '任务执行失败'
    failedExecutionId.value = cause instanceof ApiError
      && typeof cause.data === 'object'
      && cause.data !== null
      && 'execution_id' in cause.data
      ? String(cause.data.execution_id)
      : ''
  } finally {
    loading.value = false
    await nextTick()
    composer.value?.focus()
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
    try { await api.upload(file) } catch (cause) {
      error.value = cause instanceof Error ? cause.message : `上传 ${file.name} 失败`
      break
    }
  }
  await refreshDocuments()
  uploading.value = false
  input.value = ''
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
  threadId.value += 1
  messages.value = []
  error.value = ''
  failedExecutionId.value = ''
  task.value = ''
  composer.value?.focus()
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
  mobileQuery.addEventListener('change', updateViewport)
  void refreshDocuments()
  void checkConnection()
})

onBeforeUnmount(() => mobileQuery.removeEventListener('change', updateViewport))
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand-row">
        <div class="brand-mark"><BookOpen :size="17" :stroke-width="2.2" /></div>
        <span class="brand-name">知序</span>
      </div>

      <button class="new-chat" @click="newConversation"><MessageSquarePlus :size="16" />新建对话 <kbd>⌘ K</kbd></button>

      <div class="nav-label">工作区</div>
      <nav class="primary-nav" aria-label="工作区导航">
        <button class="nav-item active"><FolderOpen :size="16" /><span>知识库</span><span class="nav-count">{{ documents.length }}</span></button>
        <button class="nav-item" @click="newConversation"><MessageSquarePlus :size="16" /><span>新建对话</span></button>
      </nav>

      <div class="history-heading"><span>最近对话</span></div>
      <div class="history-list">
        <div v-if="messages.length" class="history-item selected"><span class="history-dot"></span><span class="truncate">{{ messages[0]?.text }}</span></div>
        <div v-else class="empty-history">还没有对话</div>
      </div>

      <div class="sidebar-bottom">
        <div class="local-status"><span :class="['status-dot', { online: connected }]"></span><span>{{ connected ? '本地服务已连接' : '服务未连接' }}</span><ShieldCheck :size="14" /></div>
        <div class="profile-row"><div class="avatar">知</div><span class="profile-name">本地工作区</span></div>
      </div>
    </aside>

    <main class="main-panel">
      <header class="topbar">
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

          <div v-else class="conversation" :key="threadId">
            <article v-for="message in messages" :key="message.id" :class="['message-row', message.role]">
              <div v-if="message.role === 'assistant'" class="assistant-avatar"><Sparkles :size="15" /></div>
              <div class="message-content">
                <div v-if="message.role === 'user'" class="user-message">{{ message.text }}</div>
                <template v-else>
                  <div class="answer-meta"><span class="answer-brand">知序</span><span v-if="message.result?.mode === 'agent'" class="mode-label"><Activity :size="12" />任务执行</span><span v-else class="mode-label rag-label"><Search :size="12" />知识检索</span></div>
                  <div class="answer-text">{{ message.text }}</div>
                  <div v-if="message.result?.mode === 'agent' && message.result.plan?.length" class="plan-block">
                    <div class="block-heading"><span>执行计划</span><span class="step-progress">{{ message.result.steps?.length ?? 0 }}/{{ message.result.plan.length }} 步完成</span></div>
                    <div v-for="(step, index) in message.result.plan" :key="step" class="plan-step">
                      <div :class="['step-check', { done: index < (message.result.steps?.length ?? 0) }]"><Check v-if="index < (message.result.steps?.length ?? 0)" :size="12" /></div>
                      <div class="step-detail"><span class="step-title">{{ step }}</span><span v-if="message.result.steps?.[index]" class="step-result">{{ message.result.steps[index]?.result }}</span></div>
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
            <div v-if="loading" class="loading-row"><div class="assistant-avatar"><Sparkles :size="15" /></div><LoaderCircle class="spin" :size="17" /><span>正在分析请求并执行任务…</span></div>
          </div>

          <div class="composer-area">
            <div v-if="error" class="error-banner"><span>{{ error }}</span><button v-if="failedExecutionId" class="resume-button" :disabled="loading" @click="resumeTask">恢复任务</button><button class="icon-button" aria-label="关闭错误" @click="error = ''; failedExecutionId = ''"><X :size="15" /></button></div>
            <div class="composer">
              <textarea ref="composer" v-model="task" rows="1" placeholder="向你的知识库提问，或描述一项需要完成的任务…" :disabled="loading" @keydown="onComposerKeydown"></textarea>
              <div class="composer-bottom">
                <div class="composer-tools">
                  <input ref="fileInput" type="file" accept=".pdf,.docx,.pptx,.xlsx,.md" multiple hidden @change="uploadFiles" />
                  <button class="tool-button" :disabled="uploading" title="上传文档" @click="fileInput?.click()"><Upload v-if="!uploading" :size="15" /><LoaderCircle v-else class="spin" :size="15" /><span>{{ uploading ? '正在导入' : '添加资料' }}</span></button>
                  <span class="tool-divider"></span><span class="composer-hint"><kbd>↵</kbd> 发送 <span>·</span> <kbd>⇧ ↵</kbd> 换行</span>
                </div>
                <button class="send-button" :disabled="!task.trim() || loading" aria-label="发送" title="发送" @click="sendTask"><ArrowUp :size="17" /></button>
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
              <div class="panel-heading"><div><span class="panel-kicker">任务运行时</span><h2>最近执行</h2></div><span v-if="latestRun" class="run-status complete"><CheckCircle2 :size="13" />已完成</span><span v-else class="run-status idle"><span class="status-dot"></span>待命</span></div>
              <template v-if="latestRun">
                <div class="run-task">{{ latestRun.answer }}</div>
                <div class="run-stats"><span>{{ latestRun.plan?.length ?? 0 }} 个步骤</span><span>{{ latestRun.sources?.length ?? 0 }} 个来源</span></div>
                <div v-for="step in latestRun.steps" :key="step.index" class="run-step"><span class="mini-check"><Check :size="10" /></span><span class="truncate">{{ step.task }}</span></div>
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
