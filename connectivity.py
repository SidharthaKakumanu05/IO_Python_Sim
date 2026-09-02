import cupy as cp
import numpy as np


def build_connectivity(cfg):
    N_PF, N_PKJ, N_IO, N_DCN = cfg["N_PF"], cfg["N_PKJ"], cfg["N_IO"], cfg["N_DCN"]
    rng = np.random.default_rng(cfg["seed"])
    graph = {}

    pf_conn = N_PF // 4
    graph["PF_to_PKJ"] = {
        "pre_idx": cp.asarray(rng.integers(0, N_PF, size=pf_conn * N_PKJ, dtype=np.int32)),
        "post_idx": cp.asarray(np.repeat(np.arange(N_PKJ, dtype=np.int32), pf_conn)),
    }

    pkj_per_io = N_PKJ // N_IO
    io_pre = np.repeat(np.arange(N_IO, dtype=np.int32), pkj_per_io)
    io_post = np.arange(N_IO * pkj_per_io, dtype=np.int32)
    graph["IO_to_PKJ"] = {
        "pre_idx": cp.asarray(io_pre),
        "post_idx": cp.asarray(io_post),
    }
    graph["pkj_per_io"] = pkj_per_io

    pkj_conn = N_PKJ // 8
    graph["PKJ_to_DCN"] = {
        "pre_idx": cp.asarray(rng.integers(0, N_PKJ, size=pkj_conn * N_DCN, dtype=np.int32)),
        "post_idx": cp.asarray(np.repeat(np.arange(N_DCN, dtype=np.int32), pkj_conn)),
    }

    graph["DCN_to_IO"] = {
        "pre_idx": cp.asarray(np.tile(np.arange(N_DCN, dtype=np.int32), N_IO)),
        "post_idx": cp.asarray(np.repeat(np.arange(N_IO, dtype=np.int32), N_DCN)),
    }

    perm = rng.permutation(N_IO)
    if N_IO % 2 == 1:
        perm = perm[:-1]
    graph["IO_IO"] = {
        "pre_idx": cp.asarray(perm[0::2].astype(np.int32)),
        "post_idx": cp.asarray(perm[1::2].astype(np.int32)),
    }

    return graph
