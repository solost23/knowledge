<script setup lang="ts">
import {
  BookOpen,
  Check,
  CheckCircle2,
  FileText,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from 'lucide-vue-next'
import type { AskResult, KnowledgeDocument, SourceChunk } from '../types/api'

defineProps<{
  selectedSource: SourceChunk | null
  documents: KnowledgeDocument[]
  documentCount: number
  filter: string
  uploading: boolean
  latestRun?: AskResult
  isMobile: boolean
  showMobileDetails: boolean
}>()

const emit = defineEmits<{
  clearSource: []
  'update:filter': [value: string]
  pickFiles: []
  refreshDocuments: []
  removeDocument: [document: KnowledgeDocument]
}>()

function formatDate(value: string) {
  const date = new Date(value.replace(' ', 'T'))
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric' }).format(date)
}
</script>

<template>
  <aside :class="['details-panel', { 'mobile-open': isMobile && showMobileDetails }]">
    <section v-if="selectedSource" class="source-inspector">
      <div class="panel-heading">
        <div><span class="panel-kicker">来源片段</span><h2>{{ selectedSource.source }}</h2></div>
        <button class="icon-button" aria-label="关闭来源" @click="emit('clearSource')"><X :size="16" /></button>
      </div>
      <div class="source-preview">{{ selectedSource.content }}</div>
    </section>

    <template v-else>
      <section class="details-section knowledge-section">
        <div class="panel-heading">
          <div><span class="panel-kicker">资料空间</span><h2>知识库</h2></div>
          <button class="icon-button" :disabled="uploading" title="添加文档" aria-label="添加文档" @click="emit('pickFiles')"><Plus :size="17" /></button>
        </div>
        <div class="doc-summary">
          <div class="summary-icon"><BookOpen :size="16" /></div>
          <div><strong>{{ documentCount }}</strong><span>份已索引文档</span></div>
          <button class="icon-button refresh-button" title="刷新列表" aria-label="刷新列表" @click="emit('refreshDocuments')"><RefreshCw :size="14" /></button>
        </div>
        <label class="doc-search"><Search :size="14" /><input :value="filter" placeholder="筛选文档" @input="emit('update:filter', ($event.target as HTMLInputElement).value)" /></label>
        <div class="document-list">
          <div v-for="doc in documents" :key="doc.id" class="document-item">
            <div class="document-icon"><FileText :size="15" /></div>
            <div class="document-info"><span class="document-name" :title="doc.name">{{ doc.name }}</span><span class="document-date">{{ formatDate(doc.created_at) }}</span></div>
            <button class="icon-button delete-button" :title="`删除 ${doc.name}`" :aria-label="`删除 ${doc.name}`" @click="emit('removeDocument', doc)"><Trash2 :size="14" /></button>
          </div>
          <div v-if="documents.length === 0" class="document-empty">{{ filter ? '没有匹配的文档' : '上传文档后，这里会显示索引' }}</div>
        </div>
        <button class="upload-drop" :disabled="uploading" @click="emit('pickFiles')"><Upload :size="15" /><span>{{ uploading ? '正在导入…' : '导入文档' }}</span><span class="upload-types">PDF · DOCX · PPTX · XLSX · MD · 文本 · 结构化 · 代码</span></button>
      </section>

      <section class="details-section run-section">
        <div class="panel-heading">
          <div><span class="panel-kicker">任务运行时</span><h2>最近执行</h2></div>
          <span v-if="latestRun?.status === 'running'" class="run-status running"><LoaderCircle class="spin" :size="13" />执行中</span>
          <span v-else-if="latestRun?.status === 'failed'" class="run-status failed"><X :size="13" />已中断</span>
          <span v-else-if="latestRun" class="run-status complete"><CheckCircle2 :size="13" />已完成</span>
          <span v-else class="run-status idle"><span class="status-dot"></span>待命</span>
        </div>
        <template v-if="latestRun">
          <div class="run-task">{{ latestRun.answer || latestRun.task }}</div>
          <div v-if="latestRun.status === 'running' && latestRun.current_action" class="run-current-action">{{ latestRun.current_action }}</div>
          <div class="run-stats"><span>{{ latestRun.plan?.length ?? 0 }} 个步骤</span><span>{{ latestRun.sources?.length ?? 0 }} 个来源</span></div>
          <div v-for="(step, index) in latestRun.plan" :key="step" class="run-step">
            <span :class="['mini-check', { pending: index >= (latestRun.current_step ?? latestRun.steps?.length ?? 0) }]"><Check v-if="index < (latestRun.current_step ?? latestRun.steps?.length ?? 0)" :size="10" /><LoaderCircle v-else-if="latestRun.active_step === index + 1" class="spin" :size="10" /></span>
            <span class="truncate">{{ step }}</span>
          </div>
        </template>
        <div v-else class="run-empty">提交复杂任务后，执行计划与步骤结果会显示在这里。</div>
      </section>

      <div class="security-note"><ShieldCheck :size="15" /><div><strong>本地优先</strong><span>只有模型请求会发送任务文本；索引和执行记录保留在此设备。</span></div></div>
    </template>
  </aside>
</template>
