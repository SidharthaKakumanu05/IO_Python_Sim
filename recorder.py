import numpy as np
import cupy as cp


class Recorder:
    def __init__(self, n_steps, pop_sizes, log_stride):
        self.log_stride = log_stride
        self.n_bins = n_steps // log_stride
        self.bins = {
            name: cp.zeros((self.n_bins, size), dtype=cp.float32)
            for name, size in pop_sizes.items()
        }
        self.spikes = None

    def log(self, step, pop_name, spikes):
        bin_idx = step // self.log_stride
        if bin_idx >= self.n_bins:
            return
        self.bins[pop_name][bin_idx] += spikes

    def finalize_npz(self, path, **extra):
        self.spikes = {
            name: cp.asnumpy(arr).astype(np.uint16)
            for name, arr in self.bins.items()
        }
        data = {f"{pop}_spikes": arr for pop, arr in self.spikes.items()}
        data.update(extra)
        np.savez(path, **data)
