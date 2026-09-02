import cupy as cp


def build_gap_partner(N, pre_idx, post_idx):
    partner = cp.arange(N, dtype=cp.int32)
    partner[pre_idx] = post_idx
    partner[post_idx] = pre_idx
    return partner


_gap_kernel = cp.ElementwiseKernel(
    "raw float32 V, int32 partner, float32 g_gap",
    "float32 I_gap",
    "I_gap = g_gap * (V[partner] - V[i])",
    "gap_kernel",
)


def gap_junction_current(V, partner, g_gap):
    return _gap_kernel(V, partner, g_gap)
