import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import * as vue from 'vue'

function evaluate(source, imports = {}, globals = {}) {
  const output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  const exports = {}
  new Function('require', 'exports', ...Object.keys(globals), output)((id) => imports[id] ?? {}, exports, ...Object.values(globals))
  return exports
}
const source = (file) => readFileSync(new URL(file, import.meta.url), 'utf8')
const http = evaluate(source('./src/api/http.ts'))
const tracking = evaluate(source('./src/api/tracking.ts'), { './http': http })
const deferred = () => {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const response = (data) => ({ success: true, data })
function storage(initial = {}) {
  const values = new Map(Object.entries(initial))
  return { getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) }
}
function workspace(api, initialStorage = {}, realWait = false) {
  const hooks = {}
  const localStorage = storage(initialStorage)
  const script = source('./src/App.vue').match(/<script setup lang="ts">([\s\S]*?)<\/script>/)[1]
  const exposed = 'selectConversation, newConversation, sendTask, messages, activeConversationId, task, loading, error, trackRun, reconnectRun, stopWaiting, uploadFiles, resumeDocumentImport, uploading, pendingUploadId, uploadBusy, lifetime, resumeTask, failedExecutionId'
  const result = evaluate(`${script}\nexport { ${exposed} }`, {
    vue: { ...vue, watch: () => {}, onMounted: (fn) => { hooks.mount = fn }, onBeforeUnmount: (fn) => { hooks.unmount = fn } },
    'markdown-it': { default: class {} },
    './api/client': { api: { conversations: async () => response([]), documents: async () => response([]), ...api }, ApiError: http.ApiError },
    './api/tracking': { ...tracking, waitForPoll: realWait ? tracking.waitForPoll : async (_, signal) => signal.throwIfAborted() },
  }, {
    window: { matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }), clearInterval, setTimeout, setInterval },
    localStorage,
  })
  return { ...result, hooks, localStorage }
}
const conversation = (id) => response({ id, messages: [{ id: 1, role: 'user', content: id }] })
const running = () => ({ execution_id: 'run-1', status: 'running', answer: '', mode: 'agent' })

test('旧会话成功或失败响应不能覆盖最新选择', async () => {
  for (const rejectOld of [false, true]) {
    const old = deferred()
    const app = workspace({ conversation: (id) => id === 'old' ? old.promise : Promise.resolve(conversation(id)) })
    const pending = app.selectConversation('old')
    await app.selectConversation('new')
    if (rejectOld) old.reject(new Error('旧错误'))
    else old.resolve(conversation('old'))
    await pending
    assert.equal(app.activeConversationId.value, 'new')
    assert.equal(app.messages.value[0].text, 'new')
    assert.equal(app.error.value, '')
  }
})

test('新建对话使在途响应失效，发送使用同步捕获的上下文', async () => {
  const old = deferred()
  let sentTo
  const app = workspace({
    conversation: () => old.promise,
    ask: async (_, id) => { sentTo = id; return response({ answer: '回答', status: 'completed' }) },
  })
  const selection = app.selectConversation('old')
  app.newConversation()
  old.resolve(conversation('old'))
  await selection
  assert.equal(app.activeConversationId.value, null)
  assert.equal(app.messages.value.length, 0)
  const late = deferred()
  const app2 = workspace({ conversation: () => late.promise, ask: async (_, id) => { sentTo = id; return response({ answer: '回答', status: 'completed' }) } })
  app2.activeConversationId.value = 'current'
  const selecting = app2.selectConversation('other')
  app2.task.value = '问题'
  const sending = app2.sendTask()
  late.resolve(conversation('other'))
  await Promise.all([sending, selecting])
  assert.equal(sentTo, 'current')
  assert.equal(app2.activeConversationId.value, 'current')
  assert.deepEqual(app2.messages.value.map((m) => m.text), ['问题', '回答'])
})

test('永久错误立即停止，临时错误有限重试，重新连接不重新执行', async () => {
  for (const [status, expected] of [[404, 1], [401, 1], [500, 5], [429, 5]]) {
    let calls = 0
    let broken = true
    const app = workspace({ agentRun: async (id) => {
      assert.equal(id, 'run-1')
      calls++
      if (broken) throw new http.ApiError('连接失败', null, status)
      return response({ ...running(), status: 'completed', answer: '完成' })
    } })
    const message = { id: 1, role: 'assistant', text: '', result: running() }
    await app.reconnectRun(message)
    assert.equal(calls, expected)
    assert.equal(app.loading.value, false)
    assert.equal(message.result.status, 'running')
    broken = false
    await app.reconnectRun(message)
    assert.equal(calls, expected + 1)
    assert.equal(message.text, '完成')
  }
})

test('停止等待及卸载取消轮询，保留后台 running 状态', async () => {
  for (const unmount of [false, true]) {
    const app = workspace({ agentRun: () => { throw new Error('不应发起查询') } }, {}, true)
    const message = { result: running(), text: '' }
    const pending = app.reconnectRun(message)
    if (unmount) app.hooks.unmount()
    else app.stopWaiting()
    await pending
    assert.equal(app.loading.value, false)
    assert.equal(message.result.status, 'running')
  }
})

test('上传恢复在首个请求前加锁，失败保留恢复 key 并拒绝并发上传', async () => {
  const job = deferred()
  let uploads = 0
  const key = 'knowledge.activeDocumentUpload'
  const app = workspace({ uploadTask: () => job.promise, upload: () => { uploads++ } }, { [key]: 'job-a' })
  const resume = app.resumeDocumentImport()
  assert.equal(app.uploading.value, true)
  await app.resumeDocumentImport()
  await app.uploadFiles({ target: { files: [{ name: 'b.txt' }], value: 'b' } })
  assert.equal(uploads, 0)
  job.reject(new http.ApiError('暂时断网', null, 503))
  await resume
  assert.equal(app.uploading.value, false)
  assert.equal(app.uploadBusy.value, true)
  assert.equal(app.localStorage.getItem(key), 'job-a')
})

test('上传函数防重入，旧任务完成不能清除另一个任务的恢复 key', async () => {
  const upload = deferred()
  let uploads = 0
  const key = 'knowledge.activeDocumentUpload'
  const app = workspace({ upload: () => { uploads++; return upload.promise } })
  const event = () => ({ target: { files: [{ name: 'a.txt' }], value: 'a' } })
  const first = app.uploadFiles(event())
  await app.uploadFiles(event())
  assert.equal(uploads, 1)
  upload.resolve(response({ id: 'job-a', filename: 'a.txt', status: 'completed', progress: 100, stage: '完成' }))
  await first
  assert.equal(app.uploading.value, false)
  assert.equal(app.pendingUploadId.value, '')
  const job = deferred()
  const restoring = workspace({ uploadTask: () => job.promise }, { [key]: 'job-a' })
  const pending = restoring.resumeDocumentImport()
  restoring.localStorage.setItem(key, 'job-b')
  job.resolve(response({ id: 'job-a', filename: 'a.txt', status: 'completed', progress: 100, stage: '完成' }))
  await pending
  assert.equal(restoring.localStorage.getItem(key), 'job-b')
  assert.equal(restoring.pendingUploadId.value, 'job-b')
  assert.equal(restoring.uploadBusy.value, true)
})

test('停止等待会取消在途状态请求并立即解锁', async () => {
  const started = deferred()
  let cancelled = false
  const app = workspace({ agentRun: (_, signal) => new Promise((_, reject) => {
    signal.addEventListener('abort', () => { cancelled = true; reject(signal.reason) }, { once: true })
    started.resolve()
  }) })
  const message = { result: running(), text: '' }
  const pending = app.reconnectRun(message)
  await started.promise
  app.stopWaiting()
  await pending
  assert.equal(cancelled, true)
  assert.equal(app.loading.value, false)
  assert.equal(message.result.status, 'running')
})

test('同步恢复超时或临时故障后保留原 ID，只重连查询而不重复执行', async () => {
  for (const failure of [new DOMException('超时', 'TimeoutError'), new TypeError('网络中断'), new http.ApiError('网关超时', null, 504)]) {
    let resumes = 0
    let queries = 0
    const app = workspace({
      resume: async (id) => { assert.equal(id, 'run-1'); resumes++; throw failure },
      agentRun: async (id) => { assert.equal(id, 'run-1'); queries++; return response({ ...running(), status: 'completed', answer: '恢复完成' }) },
    })
    app.failedExecutionId.value = 'run-1'
    await app.resumeTask()
    assert.equal(app.loading.value, false)
    assert.equal(app.failedExecutionId.value, '')
    const message = app.messages.value[0]
    assert.equal(message.result.execution_id, 'run-1')
    assert.equal(message.result.status, 'running')
    await app.resumeTask()
    await app.reconnectRun(message)
    assert.equal(resumes, 1)
    assert.equal(queries, 1)
    assert.equal(message.text, '恢复完成')
  }
})

test('所有上传 UI 入口共用互斥状态', () => {
  const app = source('./src/App.vue')
  const details = source('./src/components/WorkspaceDetails.vue')
  assert.match(app, /multiple hidden :disabled="uploadBusy"/)
  assert.match(app, /:disabled="uploadBusy" title="上传文档" @click="pickFiles"/)
  assert.match(app, /:uploading="uploadBusy"/)
  assert.match(app, /@pick-files="pickFiles"/)
  assert.match(details, /:disabled="uploading" title="添加文档"/)
  assert.match(details, /class="upload-drop" :disabled="uploading"/)
})

test('HTTP 错误保留状态码，超时与外部取消会中止 fetch', async () => {
  const originalFetch = globalThis.fetch
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ success: false, message: '不存在' }), { status: 404 })
    await assert.rejects(http.request('/test'), (cause) => cause instanceof http.ApiError && cause.status === 404)
    let aborted = false
    globalThis.fetch = (_, { signal }) => new Promise((_, reject) => {
      signal.addEventListener('abort', () => { aborted = true; reject(signal.reason) }, { once: true })
    })
    await assert.rejects(http.request('/test', undefined, 5), { name: 'TimeoutError' })
    assert.equal(aborted, true)
    const controller = new AbortController()
    const request = http.request('/test', { signal: controller.signal })
    controller.abort()
    await assert.rejects(request, { name: 'AbortError' })
  } finally { globalThis.fetch = originalFetch }
})

test('HTTP 超时也覆盖响应体读取，退避有上限', async () => {
  const originalFetch = globalThis.fetch
  try {
    globalThis.fetch = async (_, { signal }) => ({ ok: true, status: 200, json: () => new Promise((_, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    }) })
    await assert.rejects(http.request('/test', undefined, 5), { name: 'TimeoutError' })
    assert.deepEqual([0, 1, 2, 3, 4].map(tracking.retryDelay), [1000, 2000, 4000, 8000, 8000])
  } finally { globalThis.fetch = originalFetch }
})
