import os
import numpy as np
import matplotlib.pyplot as plt
from config import get_config

COLORS = {"PF": "green", "IO": "purple", "PKJ": "blue", "DCN": "red"}


def plot_raster(spikes, dt, log_stride, title, color, fname, outdir, max_neurons=200):
    T, N = spikes.shape
    if N > max_neurons:
        sel = np.random.default_rng(0).choice(N, size=max_neurons, replace=False)
        spikes = spikes[:, sel]
        N = max_neurons

    t = np.arange(T) * dt * log_stride
    freqs = spikes.sum(axis=0) / (T * dt * log_stride)

    fig, ax = plt.subplots(figsize=(12, max(3, N * 0.15)))
    for n in range(N):
        spk_times = t[spikes[:, n] > 0]
        if spk_times.size:
            ax.vlines(spk_times, n, n + 0.9, color=color, linewidth=0.6)

    ax.set_title(f"{title} (mean rate {freqs.mean():.2f} Hz)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


def plot_weights(t, mean, std, fname, outdir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, mean, color="darkgoldenrod", label="mean weight")
    ax.fill_between(t, mean - std, mean + std, color="darkgoldenrod", alpha=0.2, label="+/- 1 std")
    ax.set_title("PF->PKJ plastic weight multiplier over time", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Weight multiplier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close(fig)


def analyze(npz_path, outdir="analysis_outputs"):
    os.makedirs(outdir, exist_ok=True)
    cfg = get_config()
    dt, log_stride = cfg["dt"], cfg["log_stride"]

    data = np.load(npz_path)
    for pop, fname in [("PF", "pf_raster.png"), ("IO", "io_raster.png"),
                        ("PKJ", "pkj_raster.png"), ("DCN", "dcn_raster.png")]:
        key = f"{pop}_spikes"
        if key in data:
            plot_raster(data[key], dt, log_stride, f"{pop} raster", COLORS[pop], fname, outdir)
            print(f"Saved {fname}")

    if "weight_t" in data:
        plot_weights(data["weight_t"], data["weight_mean"], data["weight_std"], "pf_pkj_weight.png", outdir)
        print("Saved pf_pkj_weight.png")

    print(f"Plots saved to {outdir}/")


if __name__ == "__main__":
    analyze("cbm_py_output.npz")
