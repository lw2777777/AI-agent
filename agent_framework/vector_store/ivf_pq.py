"""
手写 IVF-PQ (Inverted File + Product Quantization)

参考: Jégou et al., 2011. Product Quantization for Nearest Neighbor Search.

IVF:
- 用 K-means 把 N 个向量聚成 nlist 个簇
- 查询时只搜最近的 nprobe 个簇

PQ:
- 把 D 维切成 M 段，每段 D/M 维
- 每段独立 K-means，用簇心 ID（8bit）代替原始 float32
- 内存压缩比 = 32 / (M * 8 / D) ... 简单说就是 32x 量级

ADC (Asymmetric Distance Computation):
- query 不量化，预先算 query 每段到该段所有簇心的距离表
- 查表累加即可，避免每次算浮点距离
"""
import heapq
from typing import List, Tuple, Optional, Dict
import numpy as np
from .base import VectorStore


class IVFPQStore(VectorStore):
    def __init__(self, dim: int, nlist: int = 256, nprobe: int = 8,
                 M: int = 8, nbits: int = 8, seed: int = 42):
        """
        dim:    向量维度，必须能被 M 整除
        nlist:  IVF 簇数
        nprobe: 查询时搜索的簇数
        M:      PQ 子空间数（段数）
        nbits:  每段簇心 ID 的位数（8 → 256 个簇心）
        """
        assert dim % M == 0, "dim 必须能被 M 整除"
        self.dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.nbits = nbits
        self.Ks = 2 ** nbits              # 每个子空间的簇心数
        self.sub_dim = dim // M
        self._rng = np.random.default_rng(seed)

        # IVF
        self.centroids: Optional[np.ndarray] = None     # (nlist, D)
        self.inverted: List[List[int]] = []             # 每个簇的向量 idx

        # PQ
        self.pq_codebooks: List[np.ndarray] = []        # M 个 (Ks, sub_dim)
        self.codes: Optional[np.ndarray] = None         # (N, M) uint8

        # 原始向量（仅用于构建，生产可丢）
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        self._raw: List[np.ndarray] = []

    # ---------------- K-means ----------------
    def _kmeans(self, X: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
        """简化 K-means，返回 (k, D) 簇心。"""
        n = len(X)
        idx = self._rng.choice(n, size=k, replace=False)
        centroids = X[idx].copy()
        for _ in range(iters):
            # 分配
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            # 更新
            new_c = np.zeros_like(centroids)
            for j in range(k):
                mask = assign == j
                if mask.any():
                    new_c[j] = X[mask].mean(0)
                else:
                    new_c[j] = X[self._rng.integers(n)]
            if np.allclose(new_c, centroids, atol=1e-5):
                break
            centroids = new_c
        return centroids

    # ---------------- 构建 ----------------
    def add(self, ids, vectors, metadata=None):
        vectors = self.normalize(np.asarray(vectors, dtype=np.float32))
        metadata = metadata or [{}] * len(ids)

        # 1. 训练 IVF 簇心
        self.centroids = self._kmeans(vectors, self.nlist)

        # 2. 分配给倒排表
        d = ((vectors[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        self.inverted = [[] for _ in range(self.nlist)]
        for i, c in enumerate(assign):
            self.inverted[c].append(i)

        # 3. 训练 PQ codebook
        self.pq_codebooks = []
        for m in range(self.M):
            sub = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            cb = self._kmeans(sub, self.Ks, iters=10)
            self.pq_codebooks.append(cb.astype(np.float32))

        # 4. 编码
        codes = np.zeros((len(vectors), self.M), dtype=np.uint8)
        for m in range(self.M):
            sub = vectors[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            cb = self.pq_codebooks[m]
            dist = ((sub[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
            codes[:, m] = dist.argmin(1).astype(np.uint8)
        self.codes = codes

        self.ids.extend(ids)
        self.metadata.extend(metadata)
        self._raw.extend(vectors)

    # ---------------- ADC 距离表 ----------------
    def _adc_table(self, q: np.ndarray) -> np.ndarray:
        """(M, Ks): query 每段到该段所有簇心的距离。"""
        table = np.zeros((self.M, self.Ks), dtype=np.float32)
        for m in range(self.M):
            sub_q = q[m * self.sub_dim:(m + 1) * self.sub_dim]
            cb = self.pq_codebooks[m]
            table[m] = ((cb - sub_q) ** 2).sum(1)
        return table

    # ---------------- 搜索 ----------------
    def search(self, query, k, filter=None):
        if self.codes is None or self.centroids is None:
            return []
        q = self.normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]

        # 1. 找最近的 nprobe 个簇
        cd = ((self.centroids - q) ** 2).sum(1)
        probe = np.argsort(cd)[:self.nprobe]

        # 2. ADC 查表
        table = self._adc_table(q)

        # 3. 在候选簇里累加距离
        cand: List[Tuple[float, int]] = []
        for c in probe:
            for i in self.inverted[c]:
                d = float(sum(table[m, self.codes[i, m]] for m in range(self.M)))
                cand.append((d, i))

        # 4. 取 Top-K
        cand.sort(key=lambda x: x[0])
        results = []
        for d, i in cand:
            if filter and not all(self.metadata[i].get(kk) == vv
                                  for kk, vv in filter.items()):
                continue
            results.append((self.ids[i], 1.0 - d))
            if len(results) >= k:
                break
        return results

    def delete(self, ids):
        drop = set(ids)
        keep = [i for i, _id in enumerate(self.ids) if _id not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.metadata = [self.metadata[i] for i in keep]
        self._raw = [self._raw[i] for i in keep]
        self.codes = self.codes[keep]
        # 重建倒排表
        self.inverted = [[] for _ in range(self.nlist)]
        if self.centroids is not None and len(self._raw):
            X = np.stack(self._raw)
            d = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
            assign = d.argmin(1)
            for i, c in enumerate(assign):
                self.inverted[c].append(i)

    def __len__(self):
        return len(self.ids)