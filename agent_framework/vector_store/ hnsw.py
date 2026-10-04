"""
手写 HNSW (Hierarchical Navigable Small World)

参考: Malkov & Yashunin, 2018. https://arxiv.org/abs/1603.09320

关键结构:
- 多层图，每层是 NSW 图；节点层级按指数衰减随机分配
- 第 0 层每个节点最多 2*M 个邻居，上层最多 M 个
- 插入: 从顶层贪心下降，逐层连接到 M 个最近邻
- 搜索: 从顶层贪心下降，第 0 层用 ef 宽度的候选队列

参数:
- M: 每层邻居数，越大召回越高、内存越大
- ef_construction: 构建时候选队列宽度，越大图质量越好、构建越慢
- ef_search: 查询时候选队列宽度，越大召回越高、延迟越大
"""
import heapq
import math
import random
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from .base import VectorStore


class _Node:
    __slots__ = ("id", "vec", "level", "neighbors", "metadata")

    def __init__(self, _id, vec, level, metadata):
        self.id = _id
        self.vec = vec                      # (D,) float32, 已归一化
        self.level = level                  # 最高层
        self.neighbors: List[List[int]] = [[] for _ in range(level + 1)]
        self.metadata = metadata or {}


class HNSWStore(VectorStore):
    def __init__(self, dim: int, M: int = 32,
                 ef_construction: int = 200,
                 ef_search: int = 50,
                 seed: int = 42,
                 distance: str = "cosine"):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M                    # 第 0 层邻居上限
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance = distance
        self._rng = random.Random(seed)
        self._ml = 1.0 / math.log(M)        # 层级采样参数

        self.nodes: List[_Node] = []
        self._id_to_idx: Dict[str, int] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1

    # ---------- 层级采样 ----------
    def _sample_level(self) -> int:
        r = self._rng.random()
        return int(-math.log(r) * self._ml)

    # ---------- 距离 ----------
    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        # 归一化后内积 = cosine，距离用 1 - cos
        return 1.0 - float(a @ b)

    # ---------- 贪心搜索（单层，返回最近点） ----------
    def _greedy_search_layer(self, q: np.ndarray, ep: int, layer: int) -> int:
        cur = ep
        cur_d = self._dist(q, self.nodes[cur].vec)
        improved = True
        while improved:
            improved = False
            for nb in self.nodes[cur].neighbors[layer]:
                d = self._dist(q, self.nodes[nb].vec)
                if d < cur_d:
                    cur_d, cur = d, nb
                    improved = True
        return cur

    # ---------- 带候选队列的搜索（第 0 层 / 构建用） ----------
    def _search_layer(self, q: np.ndarray, eps: List[int], layer: int,
                      ef: int) -> List[Tuple[float, int]]:
        visited: Set[int] = set(eps)
        # candidates: 最小堆 (dist, idx)；results: 最大堆 (-dist, idx)
        candidates = [(self._dist(q, self.nodes[e].vec), e) for e in eps]
        heapq.heapify(candidates)
        results = [(-d, e) for d, e in candidates]
        heapq.heapify(results)

        while candidates:
            d_c, c = heapq.heappop(candidates)
            d_far, _ = results[0]
            if d_c > -d_far:                # 当前候选比结果里最远的还远
                break
            for nb in self.nodes[c].neighbors[layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._dist(q, self.nodes[nb].vec)
                d_far, _ = results[0]
                if len(results) < ef or d < -d_far:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted([(-d, i) for d, i in results])

    # ---------- 邻居选择（简单剪枝：保留最近的 M 个） ----------
    def _select_neighbors(self, candidates, M, extend_candidates=False,
                      keep_pruned=True):
      """
    论文 Algorithm 4: SELECT-NEIGHBORS-HEURISTIC

    candidates: [(dist_to_q, idx), ...]  按距离升序
    M: 保留的邻居数
    extend_candidates: 是否把候选的邻居也拉进来（论文建议 True 效果更好）
    keep_pruned: 是否把剪掉的候选用作兜底填充

    返回: 选中的邻居 idx 列表
    """
    # 1. 可选：扩展候选集
      if extend_candidates:
          seen = {i for _, i in candidates}
          extra = []
          for _, c in candidates:
              for nb in self.nodes[c].neighbors[0]:
                  if nb not in seen:
                      seen.add(nb)
                      extra.append((self._dist(self._q_vec, self.nodes[nb].vec), nb))
          candidates = candidates + extra

    # 按到 query 的距离升序
      candidates = sorted(candidates, key=lambda x: x[0])

      selected: List[int] = []
      pruned: List[Tuple[float, int]] = []

      for d_q, c in candidates:
          if len(selected) >= M:
              break
        # 关键：c 和已选邻居中任意一个比 c 到 query 更近，就剪掉 c
          good = True
          for s in selected:
              d_cs = self._dist(self.nodes[c].vec, self.nodes[s].vec)
              if d_cs < d_q:
                  good = False
                  break
          if good:
              selected.append(c)
          else:
             pruned.append((d_q, c))

    # 2. 如果不够 M，用剪掉的兜底（论文 keep_pruned=True）
      if keep_pruned:
          pruned.sort(key=lambda x: x[0])
          for _, c in pruned:
              if len(selected) >= M:
                  break
              selected.append(c)

      return selected

    
    # ---------- 连接双向边 + 剪枝 ----------
    def _connect(self, src: int, dst: int, layer: int):
        src_nb = self.nodes[src].neighbors[layer]
        if dst not in src_nb:
            src_nb.append(dst)
        limit = self.M0 if layer == 0 else self.M
        if len(src_nb) > limit:
            self._q_vec = self.nodes[src].vec  
            cand = [(self._dist(q, self.nodes[n].vec), n) for n in src_nb]
            self.nodes[src].neighbors[layer] = self._select_neighbors(cand, limit)

    # ---------- 插入 ----------
    def add(self, ids, vectors, metadata=None):
        vectors = self.normalize(np.asarray(vectors, dtype=np.float32))
        metadata = metadata or [{}] * len(ids)
        for _id, vec, meta in zip(ids, vectors, metadata):
            self._insert_one(_id, vec, meta)

    def _insert_one(self, _id: str, vec: np.ndarray, meta: dict):
        level = self._sample_level()
        node = _Node(_id, vec, level, meta)
        idx = len(self.nodes)
        self.nodes.append(node)
        self._id_to_idx[_id] = idx

        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = level
            return

        ep = self.entry_point
        # 从顶层贪心下降到 level+1
        self._q_vec = vec
        for lc in range(self.max_level, level, -1):
            ep = self._greedy_search_layer(vec, ep, lc)

        # 从 min(level, max_level) 到 0，逐层连接
        for lc in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(vec, [ep], lc, self.ef_construction)
            neighbors = self._select_neighbors(cands, self.M0 if lc == 0 else self.M)
            for nb in neighbors:
                self._connect(idx, nb, lc)
                self._connect(nb, idx, lc)
            if cands:
                ep = cands[0][1]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = idx

    # ---------- 搜索 ----------
    def search(self, query, k, filter=None):
        if self.entry_point is None:
            return []
        q = self.normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        ep = self.entry_point
        for lc in range(self.max_level, 0, -1):
            ep = self._greedy_search_layer(q, ep, lc)
        cands = self._search_layer(q, [ep], 0, max(self.ef_search, k))

        results = []
        for d, i in cands:
            node = self.nodes[i]
            if filter and not all(node.metadata.get(kk) == vv
                                  for kk, vv in filter.items()):
                continue
            results.append((node.id, 1.0 - d))   # 距离转相似度
            if len(results) >= k:
                break
        return results

    def delete(self, ids):
        # 简易实现：标记删除，重建索引（生产环境用 tombstone）
        drop = set(ids)
        keep_nodes = [n for n in self.nodes if n.id not in drop]
        self.nodes = []
        self._id_to_idx = {}
        self.entry_point = None
        self.max_level = -1
        for n in keep_nodes:
            self._insert_one(n.id, n.vec, n.metadata)

    def __len__(self):
        return len(self.nodes)