# knowledge

基于 Flask + LangChain + ChromaDB 的私人笔记知识库，支持上传文档并通过 LLM 进行问答。

## 功能

- 支持上传 PDF、DOCX、PPTX、XLSX、Markdown 文档
- 使用 HuggingFace 向量模型（`all-mpnet-base-v2`）对文档分块并存入 ChromaDB
- 支持三种 LLM 后端：ChatGLM（本地）、ChatGPT、文心一言
- 问答结果附带来源文件名和匹配片段
- 内置 GPTCache 缓存层，相似问题直接命中缓存，减少 LLM 调用
- 笔记管理：列表、删除

## 快速启动

无需 Docker，直接本地运行：

```shell
pipenv install
python main.py
```

服务启动后 API 地址：`http://localhost:8086`

## 配置

所有配置项在 `conf/config.yaml`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `host` / `port` | 服务监听地址 | `0.0.0.0:8086` |
| `llm.name` | LLM 后端，可选 `chat_glm` / `chat_gpt` / `wen_xin` | `chat_glm` |
| `chroma.persist_directory` | ChromaDB 数据目录 | `./chroma_db` |
| `embedding.device` | 推理设备，可选 `mps` / `cpu` / `cuda` | `mps` |
| `cache.use` | 是否启用 GPTCache | `true` |

LLM 相关环境变量：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | ChatGPT 所需 |
| `WENXIN_API_KEY` / `WENXIN_SECRET_KEY` | 文心一言所需 |

## API

### 上传笔记

```
POST /upload
Content-Type: multipart/form-data

file: <文件>  # 支持 pdf、docx、pptx、xlsx、md
```

### 问答

```
GET /question?q=<问题内容>
```

返回：

```json
{
  "code": 0,
  "success": true,
  "message": "成功",
  "data": {
    "answer": "回答内容",
    "sources": [
      {
        "source": "笔记文件名.md",
        "content": "相关片段..."
      }
    ]
  }
}
```

### 笔记列表

```
GET /docs
```

### 删除笔记

```
DELETE /docs/<文件名>
```

## 测试

```shell
python -m unittest test.doc
python -m unittest test.question
```
