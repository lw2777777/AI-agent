from .ivf_pq import IVFPQStore

def build_store(kind, dim, **kwargs):
    if kind == "brute":
        return BruteForceStore(dim)
    if kind == "hnsw":
        return HNSWStore(dim, **kwargs)
    if kind == "ivf_pq":
        return IVFPQStore(dim, **kwargs)
    raise ValueError(kind)