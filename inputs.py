import cupy as cp


def poisson_spikes(N, rate_hz, dt):
    return cp.random.random(N) < (rate_hz * dt)


def draw_conductances(N, g_mean, g_std):
    g = cp.random.normal(loc=g_mean, scale=g_std, size=N).astype(cp.float32)
    return cp.maximum(g, 0.0)
