import math


def get_config():
    cfg = {}

    cfg["N_PF"] = 4096
    cfg["N_PKJ"] = 128
    cfg["N_IO"] = 16
    cfg["N_DCN"] = 8

    cfg["dt"] = 1e-3
    cfg["T_sec"] = 600
    cfg["T_steps"] = int(cfg["T_sec"] / cfg["dt"])
    cfg["seed"] = 12345

    cfg["pf_rate"] = 30.0

    cfg["pkj_params"] = dict(
        gL=30.0, EL=-65.0, Vth=-50.0, Vreset=-65.0,
        refrac_steps=int(0.002 / cfg["dt"]),
    )
    cfg["dcn_params"] = dict(
        gL=20.0, EL=-65.0, Vth=-50.0, Vreset=-65.0,
        refrac_steps=int(0.002 / cfg["dt"]),
        I_bias=400.0,
    )

    cfg["io_EL_range"] = (-62.0, -58.0)
    cfg["io_params"] = dict(
        gL=25.0, Vreset=-70.0,
        thresh_rest=-57.4, thresh_max=10.0, thresh_decay=1 - math.exp(-cfg["dt"] / 0.2),
        refrac_steps=int(0.002 / cfg["dt"]),
        noise_std=19.0,
    )
    cfg["io_gap_g"] = 3.0

    cfg["synapses"] = {
        "PF_PKJ": dict(E_rev=0.0, tau=4.15e-3, delay_steps=1),
        "IO_PKJ": dict(E_rev=-80.0, tau=2.0e-3, delay_steps=2),
        "PKJ_DCN": dict(E_rev=-80.0, tau=4.15e-3, delay_steps=2),
        "DCN_IO": dict(E_rev=-80.0, tau=0.3, delay_steps=3),
    }

    cfg["pf_g_mean"] = 0.12
    cfg["pf_g_std"] = 0.012
    cfg["io_pkj_g"] = 150.0
    cfg["pkj_dcn_g"] = 0.5
    cfg["dcn_io_g"] = 0.05

    cfg["weight_init"] = 1.0
    cfg["weight_min"] = 0.2
    cfg["weight_max"] = 3.0
    cfg["plasticity_window"] = 0.1
    cfg["ltd_frac"] = 0.1
    cfg["ltp_strength"] = 0.001
    cfg["ltd_strength"] = 0.009
    cfg["rec_weight_every_steps"] = int(0.1 / cfg["dt"])

    cfg["log_stride"] = 10

    return cfg
