# 评测（eval）

RAG 与 Agent 共用合成企业制度语料，但使用不同评测集和指标。RAG 评测回答三个问题：

1. 检索有没有把正确的文档/证据捞回来？
2. 答案有没有覆盖关键事实？
3. 换 chunk size、k、embedding 之后，是变好还是变坏？

## 目录

```
evals/
├── corpus/              # 语料（当前为合成的企业制度文档，17 篇）
├── questions.jsonl      # 评测集，一行一题
├── agent_questions.jsonl # Agent 路由与执行用例
├── agent.py             # Agent 端到端评测入口
├── agent_metrics.py     # Agent 评测指标
├── run.py               # 评测入口
├── .chroma/             # 评测专用向量库（自动生成，已 gitignore）
└── reports/             # 报告（json + md，自动生成）
```

## 运行

```bash
# 只评检索，不花 token
uv run --project backend --locked python backend/evals/run.py --no-answer

# 完整评测（检索 + 答案）
uv run --project backend --locked python backend/evals/run.py

# 换切分参数做对比
uv run --project backend --locked python backend/evals/run.py --chunk-size 300 --chunk-overlap 30 --tag chunk300
uv run --project backend --locked python backend/evals/run.py --chunk-size 800 --chunk-overlap 80 --tag chunk800
```

## 指标

| 指标 | 含义 |
|---|---|
| `doc_hit_rate` | top-k 里是否包含标注的正确文档 |
| `evidence_hit_rate` | top-k 里是否包含能回答问题的证据片段（更严格） |
| `mrr` | 正确文档排名的倒数均值，衡量排序质量 |
| `keyword_recall` | 答案覆盖标注关键词的比例 |
| `answer_accuracy` | 答案是否覆盖全部关键词（严格） |
| `abstention_rate` | 无答案问题中，模型正确表示"不知道"的比例 |
| `avg_retrieval_ms` / `avg_gen_ms` | 检索 / 生成平均耗时 |

答案指标用关键词覆盖做近似，成本低、可复现；需要更严格的语义判断时，可在此基础上加 LLM judge。

## Agent 评测

先把 `evals/corpus/` 下的文档上传到当前服务知识库，再运行：

```bash
uv run --project backend --locked python backend/evals/agent.py --limit 4
uv run --project backend --locked python backend/evals/agent.py --tag before_prompt_change
```

该脚本会真实调用模型和问答服务，因此会产生模型费用。它会报告路由准确率、Agent 完成率、计划步数是否合法、计划关键词覆盖、步骤完成率、预期工具召回和答案关键词召回。关键词指标是轻量近似，不能替代人工判断或语义评审。

每条用例会根据预期路由、执行成功、计划约束、必需工具和答案关键词输出 `PASS` / `FAIL`；报告会记录全部指标，便于在提示词或实现调整前后比较。

`agent_questions.jsonl` 每行是一条固定用例，包含 `task`、`expected_mode`，以及可选的 `plan_terms`、`answer_terms`、`expected_tools` 和 `min_plan_steps`。改变路由提示词、规划提示词或工具实现后，使用相同评测集对比报告。

## 换成自己的语料

1. 把文档（`.md` / `.txt`）放进 `evals/corpus/`，删掉 demo 文件；
2. 按下面的格式重写 `questions.jsonl`：

```json
{"id":"q01","question":"问题","gold_doc":"文件名.md","gold_evidence":"文档中能回答问题的原文片段","reference":"参考答案","keywords":["关键","词"],"category":"single"}
```

- `gold_doc`：正确来源文件名；无答案的问题填 `"__none__"`。
- `gold_evidence`：**必须**是文档中的原文子串，脚本靠它判断证据命中。
- `keywords`：参考答案里必须出现的关键词/数字，用于判断答案覆盖。
- `category`：`single` / `multi` / `unanswerable`，仅用于分组观察。

生成或修改评测集后，先跑一次 `--no-answer` 确认 `evidence_hit_rate` 不是 0——如果是 0，通常是 `gold_evidence` 和原文对不上。

## 当前 demo 语料的局限

- 文档是合成的企业制度文本，规模小，指标偏乐观，**只用于验证框架和做参数对比**。
- 语料中包含一份已废止的旧版报销制度，用来模拟真实企业知识库里的"过期文档干扰"。
- 14 篇文档中有 3 篇长文档会真正触发切分；其余文档不足 500 字，一个文档就是一个 chunk。
