import math

import cupy as cp

from coupling import gap_junction_current


class NeuronState:
    def __init__(self, N, params):
        self.N = N
        self.V = cp.full(N, params["EL"], dtype=cp.float32)
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        self.spike = cp.zeros(N, dtype=bool)


@cp.fuse()
def _lif_fused(V, refrac_timer, I_syn, dt, gL, EL, Vth, Vreset, refrac_steps):
    refrac2 = cp.maximum(refrac_timer - 1, 0)
    active = refrac2 == 0
    V_new = V + dt * active * (gL * (EL - V) + I_syn)
    spike = V_new >= Vth
    V_out = cp.where(spike, Vreset, V_new)
    refrac_out = cp.where(spike, refrac_steps, refrac2)
    return V_out, refrac_out, spike


def lif_step(state, I_syn, dt, params):
    V, refrac, spike = _lif_fused(
        state.V, state.refrac_timer, I_syn,
        cp.float32(dt), cp.float32(params["gL"]), cp.float32(params["EL"]),
        cp.float32(params["Vth"]), cp.float32(params["Vreset"]), cp.int32(params["refrac_steps"]),
    )
    state.V, state.refrac_timer, state.spike = V, refrac, spike
    return state


class IOState:
    def __init__(self, N, params, EL_range, dt, seed=0):
        self.N = N
        cp.random.seed(seed)

        self.EL = cp.random.uniform(EL_range[0], EL_range[1], N).astype(cp.float32)

        self.gL = cp.float32(params["gL"])
        self.Vreset = cp.float32(params["Vreset"])
        self.thresh_rest = cp.float32(params["thresh_rest"])
        self.thresh_max = cp.float32(params["thresh_max"])
        self.thresh_decay = cp.float32(params["thresh_decay"])
        self.refrac_steps = params["refrac_steps"]

        self.noise_std = float(params.get("noise_std", 0.0))
        self.noise_scale = cp.float32(self.noise_std / math.sqrt(dt)) if self.noise_std > 0 else cp.float32(0.0)
        self.zero_noise = cp.zeros(N, dtype=cp.float32)

        self.V = self.EL + cp.random.normal(0.0, 1.0, N).astype(cp.float32)
        self.thresh = cp.full(N, params["thresh_rest"], dtype=cp.float32)
        self.refrac_timer = cp.zeros(N, dtype=cp.int32)
        self.spike = cp.zeros(N, dtype=bool)


@cp.fuse()
def _io_fused(V, thresh, refrac, EL, I_gap, I_ext, noise, dt, gL, Vreset,
              thresh_rest, thresh_max, thresh_decay, noise_scale, refrac_steps):
    thresh2 = thresh + thresh_decay * (thresh_rest - thresh)
    refrac2 = cp.maximum(refrac - 1, 0)
    active = refrac2 == 0
    dV = gL * (EL - V) + I_gap + I_ext + noise_scale * noise
    V_new = V + dt * active * dV
    spike = (V_new >= thresh2) & active
    V_out = cp.where(spike, Vreset, V_new)
    refrac_out = cp.where(spike, refrac_steps, refrac2)
    thresh_out = cp.where(spike, thresh_max, thresh2)
    return V_out, thresh_out, refrac_out, spike


def io_step(state, dt, I_ext, gap_partner, g_gap):
    I_gap = gap_junction_current(state.V, gap_partner, g_gap)
    noise = cp.random.standard_normal(state.N, dtype=cp.float32) if state.noise_std > 0 else state.zero_noise

    V, thresh, refrac, spike = _io_fused(
        state.V, state.thresh, state.refrac_timer, state.EL, I_gap, I_ext, noise,
        cp.float32(dt), state.gL, state.Vreset, state.thresh_rest, state.thresh_max,
        state.thresh_decay, state.noise_scale, state.refrac_steps,
    )
    state.V, state.thresh, state.refrac_timer, state.spike = V, thresh, refrac, spike
    return state
