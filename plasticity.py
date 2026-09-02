import cupy as cp


class PFPlasticity:
    def __init__(self, pre_idx, post_idx, n_pre, pkj_per_io, weight_init, weight_min, weight_max,
                 window_steps, ltd_steps, ltp_strength, ltd_strength):
        self.pre_idx = pre_idx
        self.post_idx = post_idx
        self.pkj_per_io = pkj_per_io

        self.w = cp.full(pre_idx.size, weight_init, dtype=cp.float32)
        self.weight_min = cp.float32(weight_min)
        self.weight_max = cp.float32(weight_max)
        self.ltp_strength = cp.float32(ltp_strength)
        self.ltd_strength = cp.float32(ltd_strength)

        self.window_steps = window_steps
        self.ltd_steps = ltd_steps
        self.big_buf = cp.zeros((window_steps, n_pre), dtype=cp.float32)
        self.small_buf = cp.zeros((ltd_steps, n_pre), dtype=cp.float32)
        self.big_ptr = 0
        self.small_ptr = 0
        self.total_count = cp.zeros(n_pre, dtype=cp.float32)
        self.ltd_count = cp.zeros(n_pre, dtype=cp.float32)
        self.pkj_block = self.post_idx // self.pkj_per_io

    def step(self, pf_spike, io_spike):
        big_row = self.big_buf[self.big_ptr]
        small_row = self.small_buf[self.small_ptr]

        pf_f, self.total_count, self.ltd_count, ltp_count = _ring_update(
            pf_spike, big_row, small_row, self.total_count, self.ltd_count,
        )

        self.big_buf[self.big_ptr] = pf_f
        self.big_ptr = (self.big_ptr + 1) % self.window_steps
        self.small_buf[self.small_ptr] = pf_f
        self.small_ptr = (self.small_ptr + 1) % self.ltd_steps

        cf_active = io_spike[self.pkj_block]

        self.w = _plasticity_update(
            self.w, self.ltp_strength, ltp_count[self.pre_idx],
            self.ltd_strength, self.ltd_count[self.pre_idx], cf_active,
            self.weight_min, self.weight_max,
        )
        return self.w


@cp.fuse()
def _ring_update(pf_spike, big_row, small_row, total_count, ltd_count):
    pf_f = pf_spike.astype(cp.float32)
    new_total = total_count + pf_f - big_row
    new_ltd = ltd_count + pf_f - small_row
    ltp_count = new_total - new_ltd
    return pf_f, new_total, new_ltd, ltp_count


@cp.fuse()
def _plasticity_update(w, ltp_strength, ltp_count, ltd_strength, ltd_count, cf_active, weight_min, weight_max):
    dw = (ltp_strength * ltp_count - ltd_strength * ltd_count) * cf_active
    return cp.clip(w + dw, weight_min, weight_max)
