import cupy as cp
import numpy as np


_scatter_kernel = cp.RawKernel(r"""
extern "C" __global__
void synapse_scatter(const bool* pre_spikes, const int* pre_idx, const int* post_idx,
                      const float* w, int n_edges, float* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_edges) return;
    if (pre_spikes[pre_idx[i]]) {
        atomicAdd(&out[post_idx[i]], w[i]);
    }
}
""", "synapse_scatter")


class Synapse:
    def __init__(self, pre_idx, post_idx, w, E_rev, tau, delay_steps, n_post, dt):
        self.pre_idx = cp.asarray(pre_idx, dtype=cp.int32)
        self.post_idx = cp.asarray(post_idx, dtype=cp.int32)
        w = cp.asarray(w, dtype=cp.float32)
        self.w = cp.broadcast_to(w, self.pre_idx.shape).copy() if w.ndim == 0 else w

        self.E_rev = cp.float32(E_rev)
        self.alpha = cp.float32(np.exp(-dt / tau))
        self.delay_steps = delay_steps
        self.n_post = n_post
        self.n_edges = self.pre_idx.size

        self.block = 128
        self.grid = (self.n_edges + self.block - 1) // self.block

        self.g = cp.zeros(n_post, dtype=cp.float32)
        self.new_input = cp.zeros(n_post, dtype=cp.float32)
        self.queue = [cp.zeros(n_post, dtype=cp.float32) for _ in range(delay_steps)]
        self.qptr = 0

    def step(self, pre_spikes, V_post):
        self.new_input.fill(0)
        _scatter_kernel(
            (self.grid,), (self.block,),
            (pre_spikes, self.pre_idx, self.post_idx, self.w, self.n_edges, self.new_input),
        )

        if self.delay_steps > 0:
            due = self.queue[self.qptr]
            self.queue[self.qptr] = self.new_input
            self.new_input = due
            self.qptr = (self.qptr + 1) % self.delay_steps
        else:
            due = self.new_input

        self.g, I_syn = _synapse_output(self.g, self.alpha, due, self.E_rev, V_post)
        return I_syn


@cp.fuse()
def _synapse_output(g, alpha, due, E_rev, V_post):
    g_new = g * alpha + due
    return g_new, g_new * (E_rev - V_post)
