import os
import numpy as np
import cupy as cp
from tqdm import tqdm

from config import get_config
from neurons import NeuronState, lif_step, IOState, io_step
from synapses import Synapse
from connectivity import build_connectivity
from coupling import build_gap_partner
from inputs import poisson_spikes, draw_conductances
from plasticity import PFPlasticity
from recorder import Recorder


def run():
    cfg = get_config()
    dt = cfg["dt"]
    T_steps = cfg["T_steps"]
    N_PF, N_PKJ, N_IO, N_DCN = cfg["N_PF"], cfg["N_PKJ"], cfg["N_IO"], cfg["N_DCN"]

    graph = build_connectivity(cfg)
    pkj_per_io = graph["pkj_per_io"]

    PKJ = NeuronState(N_PKJ, cfg["pkj_params"])
    DCN = NeuronState(N_DCN, cfg["dcn_params"])
    IO = IOState(N_IO, cfg["io_params"], cfg["io_EL_range"], dt, seed=cfg["seed"])

    gap_partner = build_gap_partner(N_IO, graph["IO_IO"]["pre_idx"], graph["IO_IO"]["post_idx"])

    pf_base_g = draw_conductances(graph["PF_to_PKJ"]["pre_idx"].size, cfg["pf_g_mean"], cfg["pf_g_std"])

    window_steps = int(round(cfg["plasticity_window"] / dt))
    ltd_steps = max(1, int(round(window_steps * cfg["ltd_frac"])))
    pf_plasticity = PFPlasticity(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        n_pre=N_PF, pkj_per_io=pkj_per_io,
        weight_init=cfg["weight_init"], weight_min=cfg["weight_min"], weight_max=cfg["weight_max"],
        window_steps=window_steps, ltd_steps=ltd_steps,
        ltp_strength=cfg["ltp_strength"], ltd_strength=cfg["ltd_strength"],
    )

    pf_pkj = Synapse(
        graph["PF_to_PKJ"]["pre_idx"], graph["PF_to_PKJ"]["post_idx"],
        w=pf_base_g * pf_plasticity.w,
        n_post=N_PKJ, dt=dt, **cfg["synapses"]["PF_PKJ"],
    )
    io_pkj = Synapse(
        graph["IO_to_PKJ"]["pre_idx"], graph["IO_to_PKJ"]["post_idx"],
        w=cp.float32(cfg["io_pkj_g"]),
        n_post=N_PKJ, dt=dt, **cfg["synapses"]["IO_PKJ"],
    )
    pkj_dcn = Synapse(
        graph["PKJ_to_DCN"]["pre_idx"], graph["PKJ_to_DCN"]["post_idx"],
        w=cp.float32(cfg["pkj_dcn_g"]),
        n_post=N_DCN, dt=dt, **cfg["synapses"]["PKJ_DCN"],
    )
    dcn_io = Synapse(
        graph["DCN_to_IO"]["pre_idx"], graph["DCN_to_IO"]["post_idx"],
        w=cp.float32(cfg["dcn_io_g"]),
        n_post=N_IO, dt=dt, **cfg["synapses"]["DCN_IO"],
    )

    pop_sizes = {"PF": N_PF, "IO": N_IO, "PKJ": N_PKJ, "DCN": N_DCN}
    rec = Recorder(T_steps, pop_sizes, log_stride=cfg["log_stride"])

    weight_t, weight_mean, weight_std = [], [], []

    for step in tqdm(range(T_steps), desc="Simulating", unit="steps"):
        pf_spike = poisson_spikes(N_PF, cfg["pf_rate"], dt)

        I_pf = pf_pkj.step(pf_spike, PKJ.V)
        I_io = io_pkj.step(IO.spike, PKJ.V)
        PKJ = lif_step(PKJ, I_pf + I_io, dt, cfg["pkj_params"])

        I_pkj = pkj_dcn.step(PKJ.spike, DCN.V)
        DCN = lif_step(DCN, I_pkj + cfg["dcn_params"]["I_bias"], dt, cfg["dcn_params"])

        I_dcn = dcn_io.step(DCN.spike, IO.V)
        IO = io_step(IO, dt, I_dcn, gap_partner, cfg["io_gap_g"])

        pf_plasticity.step(pf_spike, IO.spike)
        pf_pkj.w = pf_base_g * pf_plasticity.w

        rec.log(step, "PF", pf_spike)
        rec.log(step, "IO", IO.spike)
        rec.log(step, "PKJ", PKJ.spike)
        rec.log(step, "DCN", DCN.spike)

        if step % cfg["rec_weight_every_steps"] == 0:
            weight_t.append(step * dt)
            weight_mean.append(float(pf_plasticity.w.mean()))
            weight_std.append(float(pf_plasticity.w.std()))

    out_npz = "cbm_py_output.npz"
    rec.finalize_npz(
        out_npz,
        weight_t=np.array(weight_t),
        weight_mean=np.array(weight_mean),
        weight_std=np.array(weight_std),
    )
    return {"out_npz": os.path.abspath(out_npz)}


if __name__ == "__main__":
    info = run()
    print("Saved outputs to", info["out_npz"])
