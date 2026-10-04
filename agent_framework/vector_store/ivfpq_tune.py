"""
对比暴力 / HNSW / IVF-PQ 的召回、QPS、内存。
"""
import time, sys
import numpy as np
from vector_store.brute_force import BruteForceStore
from vector_store.hnsw import HNSWStore
from vector_store.ivf_pq import IVFPQStore


def gen_data(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [f"v{i}" for i in range(n)], v


def recall(gt, pred, k):
    return len(set(gt[:k]) & set(pred[:k])) / k


def mem_mb(store):
    """粗略估算存储占用。"""
    if hasattr(store, "matrix") and store.matrix is not None:
        return store.matrix.nbytes / 1e6
    if hasattr(store, "codes") and store.codes is not None:
        return store.codes.nbytes / 1e6
    return -1


def main():
    N, D, NQ, K = 50_000, 128, 200, 10
    ids, vecs = gen_data(N, D)
    _, qs = gen_data(NQ, D, seed=999)

    brute = BruteForceStore(D)
    brute.add(ids, vecs)
    gt = [[i for i, _ in brute.search(q, K)] for q in qs]

    def run(name, store):
        t0 = time.time()
        store.add(ids, vecs)
        build = time.time() - t0
        t0 = time.time()
        recs = []
        for q, g in zip(qs, gt):
            pred = [i for i, _ in store.search(q, K)]
            recs.append(recall(g, pred, K))
        qps = NQ / (time.time() - t0)
        print(f"{name:12s} recall@10={np.mean(recs):.3f} "
              f"qps={qps:6.0f} build={build:5.1f}s mem={mem_mb(store):.1f}MB")

    run("brute", BruteForceStore(D))
    run("hnsw", HNSWStore(D, M=32, ef_search=100))

    for nprobe in [1, 4, 8, 16]:
        run(f"ivfpq-np{nprobe}",
            IVFPQStore(D, nlist=256, nprobe=nprobe, M=8))


if __name__ == "__main__":
    main()