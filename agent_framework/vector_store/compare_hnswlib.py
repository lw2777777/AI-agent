"""
对比手写 HNSW vs hnswlib：
- 相同数据、相同 M / ef
- 指标：召回率、QPS、内存
"""
import time
import numpy as np
import hnswlib
from vector_store.hnsw import HNSWStore


def gen_data(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [f"v{i}" for i in range(n)], v


def main():
    N, D, NQ, K = 50_000, 128, 200, 10
    ids, vecs = gen_data(N, D)
    q_ids, q_vecs = gen_data(NQ, D, seed=999)

    # hnswlib
    lib = hnswlib.Index(space="cosine", dim=D)
    lib.init_index(max_elements=N, ef_construction=200, M=32)
    t0 = time.time()
    lib.add_items(vecs, np.arange(N))
    lib_build = time.time() - t0
    lib.set_ef(50)

    t0 = time.time()
    lib_labels, _ = lib.knn_query(q_vecs, k=K)
    lib_qps = NQ / (time.time() - t0)

    # 手写
    mine = HNSWStore(D, M=32, ef_construction=200, ef_search=50)
    t0 = time.time()
    mine.add(ids, vecs)
    my_build = time.time() - t0

    t0 = time.time()
    my_preds = []
    for q in q_vecs:
        my_preds.append([i for i, _ in mine.search(q, K)])
    my_qps = NQ / (time.time() - t0)

    # 召回：以 hnswlib 结果为参考（也可用暴力搜索做真值）
    hit = 0
    for lib_lbl, my in zip(lib_labels, my_preds):
        hit += len(set(map(int, lib_lbl)) & set(int(x[1:]) for x in my))
    recall = hit / (NQ * K)

    print(f"hnswlib: build={lib_build:.1f}s qps={lib_qps:.0f}")
    print(f"mine   : build={my_build:.1f}s qps={my_qps:.0f}")
    print(f"recall vs hnswlib: {recall:.3f}")


if __name__ == "__main__":
    main()