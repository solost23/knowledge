# RAG 评测报告：fe_chunk500_k5

| 指标 | 值 |
|---|---|
| tag | fe_chunk500_k5 |
| chunk_size | 500 |
| chunk_overlap | 50 |
| k | 5 |
| docs | 17 |
| chunks | 18 |
| build_ms | 1666.2 |
| questions | 40 |
| answerable | 37 |
| doc_hit_rate | 1.0 |
| evidence_hit_rate | 1.0 |
| mrr | 0.8468 |
| avg_retrieval_ms | 8.8 |
| keyword_recall | 0.9865 |
| answer_accuracy | 0.973 |
| avg_gen_ms | 3130.1 |
| total_tokens | 59068 |
| abstention_rate | 1.0 |

## 检索失败


## 答案未覆盖关键词

- q32 远程办公期间可以操作生产发布吗？ → 不可以。

根据发布流程规定，禁止在远程办公环境下执行生产变更操作。
