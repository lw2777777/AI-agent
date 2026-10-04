"""
HNSW 调参实验：
- 固定数据，扫 M / ef_construction / ef_search
- 指标：召回率@10 vs 暴力搜索、QPS、内存
产出：三张 CSV + 曲线图（自己用 matplotlib 画）
"""
import time
import numpy as np
from vector_store.brute_force import BruteForceStore
from vector_store.hnsw import HNSWStore


def gen_data(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"v{i}" for i in range(n)]
    return ids, vecs


def recall_at_k(gt_ids, pred_ids, k):
    gt = set(gt_ids[:k])
    pred = set(pred_ids[:k])
    return len(gt & pred) / k


def main():
    N, D, NQ, K = 50_000, 128, 200, 10
    ids, vecs = gen_data(N, D)
    q_ids, q_vecs = gen_data(NQ, D, seed=999)

    # 基线
    brute = BruteForceStore(D)
    brute.add(ids, vecs)

    gt = []
    for q in q_vecs:
        gt.append([i for i, _ in brute.search(q, K)])

    # 扫参数
    rows = []
    for M in [8, 16, 32]:
        for efc in [100, 200]:
            for efs in [10, 50, 100, 200]:
                hnsw = HNSWStore(D, M=M, ef_construction=efc, ef_search=efs)
                t0 = time.time()
                hnsw.add(ids, vecs)
                build_t = time.time() - t0

                t0 = time.time()
                recs = []
                for q, g in zip(q_vecs, gt):
                    pred = [i for i, _ in hnsw.search(q, K)]
                    recs.append(recall_at_k(g, pred, K))
                query_t = time.time() - t0
                qps = NQ / query_t

                rows.append((M, efc, efs, np.mean(recs), qps, build_t))
                print(f"M={M} efc={efc} efs={efs} "
                      f"recall={np.mean(recs):.3f} qps={qps:.0f} "
                      f"build={build_t:.1f}s")

    # 写 CSV
    import csv
    with open("hnsw_tune.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ef_construction", "ef_search",
                    "recall@10", "qps", "build_sec"])
        w.writerows(rows)


if __name__ == "__main__":
    main()