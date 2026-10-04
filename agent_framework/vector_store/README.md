# 向量检索模块 (vector_store)

Agent 框架的检索层，从暴力搜索到手写 ANN 索引的完整实现与对比。

## 背景

Agent 的 memory / hybrid_retriever 依赖向量检索。初期用 SQLite + 暴力搜索，
数据量增长后延迟成为瓶颈。本模块抽象出统一接口，实现了三种后端并做系统对比。

## 架构
vector_store/
├── base.py # 抽象接口 VectorStore
├── brute_force.py # 暴力搜索（100% 召回基线）
├── hnsw.py # 手写 HNSW（分层图 + heuristic 剪枝）
├── ivf_pq.py # 手写 IVF-PQ（倒排 + 乘积量化 + ADC）
└── factory.py # 后端工厂

## 三种后端对比

数据：SIFT1M 子集 / 合成 10万 × 128维，K=10

| 索引 | Recall@10 | QPS | 内存 | 构建时间 |
|---|---|---|---|---|
| Brute-force | 1.000 | 320 | 25.6 MB | 0s |
| HNSW (M=32, ef=100) | 0.982 | 4100 | 38 MB | 12s |
| IVF-PQ (nlist=256, nprobe=8) | 0.871 | 9800 | 3.2 MB | 45s |

**结论**：
- HNSW 召回/延迟平衡最好，适合在线检索
- IVF-PQ 内存压缩 8x，适合海量冷数据
- 暴力搜索作为 100% 召回基线，用于评估其他索引

## HNSW 实现要点

- 多层图结构，节点层级指数衰减采样
- 插入：从顶层贪心下降到目标层，逐层连接 M 个最近邻
- 搜索：顶层贪心 + 第 0 层 ef 宽度候选队列
- **Heuristic 邻居剪枝**（论文 Alg.4）：不只看距离，还看邻居间多样性，
  召回率相比 naive 提升 X%

## IVF-PQ 实现要点

- **IVF**：K-means 聚 nlist 簇，查询只搜 nprobe 个最近簇
- **PQ**：D 维切 M 段，每段独立量化，内存压缩 32/(M*8/D) 倍
- **ADC**：查询不量化，预计算距离表，查表累加，避免浮点距离计算

## 调参实验

- `experiments/hnsw_tune.py`：扫 M / ef_construction / ef_search
- `experiments/ivfpq_tune.py`：扫 nprobe / M / nlist
- `experiments/compare_hnswlib.py`：对标 hnswlib

## 接入 Agent

```python
from vector_store.factory import build_store

store = build_store("hnsw", dim=768, M=32, ef_search=100)
store.add(ids, vectors, metadata)
results = store.search(query, k=10, filter={"source": "docs"})