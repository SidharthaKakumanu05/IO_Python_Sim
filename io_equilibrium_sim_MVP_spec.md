# PROJECT: Olivo-Cerebellar Microcircuit Simulation — IO Equilibrium & LTD-Window Hypotheses (MVP)

## 0. Context for the coding agent

This is a small, CPU-friendly Python research simulation, architecturally inspired by **CbmSim**
(Mauk Lab, UT Austin — single-compartment biologically-derived cerebellum simulator,
https://github.com/mmauk/CbmSim). It is NOT a port of CbmSim and does not need CUDA, C++,
or CbmSim's full granule-layer/mossy-fiber front end. It is a minimal circuit built to test two
specific hypotheses from an internal Mauk Lab white paper on Inferior Olive (IO) function
(see Section 1). If any connectivity ratio, population size, or dynamical detail is not specified
below, look it up in CbmSim's source if it is available on disk or in a connected repo (the
relevant modules are the ones defining Purkinje cell / DCN / IO population sizes and the
PF→PKJ, PKJ→DCN, DCN→IO, IO→PKJ connection tables). If CbmSim is not accessible, use the
defaults given here (which are pulled directly from the white paper) and clearly flag them as
configurable assumptions in the code/config rather than hard-coding guesses.

Build this iteratively: get a runnable skeleton first (random spiking, no plasticity), then add
the coincidence-detection plasticity rule, then add the equilibrium/ablation experiments.

---

## 1. Scientific hypotheses this sim must be able to test

**H1 — Triple-negative equilibrium feedback.**
The loop PKJ ⊣ DCN ⊣ IO, combined with IO ⊣ PKJ acting as a teaching signal that drives LTD at
active gr→PKJ (PF→PKJ) synapses, forms a closed negative-feedback loop that drives IO firing to
a stable equilibrium rate (~1 Hz) and thereby prevents unbounded drift of PF→PKJ synaptic
weights. Prediction: with the loop intact, PF→PKJ weights and IO rate converge and stay stable.
If the loop is opened (e.g., DCN→IO connection ablated, or IO forced to a fixed exogenous rate),
weights should drift monotonically instead of stabilizing.

**H2 — LTD/LTP window asymmetry must balance at equilibrium.**
A gr→PKJ synapse undergoes LTD if a PF spike occurred within a fixed LTD window immediately
preceding a climbing fiber (CF) event from IO; otherwise it undergoes LTP. Because the LTD
window (~100 ms) is much shorter than the average inter-CF interval at equilibrium (~1000 ms
at 1 Hz), LTD events are rare (~10% of the time) but must be ~9x larger in magnitude (δ−) than
LTP events (δ+) for weights to balance on average at equilibrium (δ+ · P(LTP) ≈ δ− · P(LTD)).
Prediction: sweeping the LTD window length changes the δ+/δ− ratio needed for zero net drift in
a simple, predictable way (ratio ≈ (1000 − window_ms) / window_ms at 1 Hz equilibrium).

**Stretch hypothesis (build only if H1/H2 work end-to-end) — Three-window scheme.**
Splitting the post-CF interval into LTD window → null (no-plasticity) window → LTP window
(rather than LTD window → LTP-everywhere-else) should let δ+ and δ− be closer in magnitude and
should slow the random-walk "forgetting" rate, at some cost to extinction learning (a longer
null window can occasionally block a needed LTP event if a spontaneous CF's null window overlaps
CS-eligible synapses — the "extinction penalty"). This is a secondary experiment; do not block
the MVP on it, but structure the plasticity module so this variant is a config flag, not a
rewrite.

Out of scope for this MVP (explicitly, per project owner): IO subthreshold oscillations and
IO-IO electrical coupling. Do not implement oscillatory conductances or gap-junction coupling.
Leave a clearly marked extension point (e.g., an `io_coupling.py` stub / TODO) since the white
paper treats these as central to a *later* phase of the project.

---

## 2. Circuit architecture (MVP)

Four populations, rate-coded or simple LIF (implementer's choice, see Section 4), connected as:

```
PF pool (Poisson spike sources)
    │  excitatory, PLASTIC, many-to-one convergent
    ▼
Purkinje cells (PKJ)            [N_PKJ]
    │  inhibitory, static
    ▼
Deep cerebellar nucleus (DCN)   [N_DCN]
    │  inhibitory, static
    ▼
Inferior olive (IO)             [N_IO]
    │  inhibitory + "teaching" (climbing fiber / CF event), static weight,
    │  but its TIMING drives the plasticity rule on PF→PKJ synapses
    └──────────────────────────────────────────────────────────▲
                                                                 │
                                    (IO output loops back onto the SAME
                                     PKJ population it modulates — see
                                     "closed loop" constraint below)
```

### 2.1 Population sizes (defaults; override via config)

Ratios below are taken directly from the white paper and should be treated as authoritative
unless CbmSim's source gives more precise figures:

- Each IO neuron's climbing fiber diverges to **8–10 PKJ cells**.
- Each DCN neuron receives convergent input from **~45 PKJ cells**.
- N_IO ≪ N_DCN ≪ N_PKJ (there are far more PKJ than DCN, and far more DCN than IO), so each
  DCN neuron must diverge to contact multiple IO neurons, and each IO neuron is inhibited by
  multiple DCN neurons.

Suggested MVP scale (small enough to run instantly on CPU, large enough to be meaningful):

- `N_PF = 500` (Poisson PF pool feeding into each PKJ's dendritic tree — this can be either
  one shared pool sampled per PKJ, or per-PKJ private pools; use a per-PKJ private pool of
  `N_PF_PER_PKJ = 500` for the MVP so H2's coincidence detection is clean and independent
  per PKJ)
- `N_PKJ = 10` per IO (so `N_PKJ_PER_IO = 8` to `10`, matching the paper; start with 8)
- `N_DCN = 2` (each DCN receives from a subset of ~45 PKJs in the full system; for the MVP,
  since we only have ~8 PKJ per IO, just make each DCN neuron receive from ALL PKJs under its
  IO group — document this simplification explicitly in code comments as a scale-down of the
  45:1 ratio, not a violation of its direction)
- `N_IO = 1` for the MVP's core experiment (single fully-closed loop, see 2.2). Support `N_IO > 1`
  as a config option for a later experiment about cross-IO drift (mentioned in the white paper as
  the reason electrical coupling matters — not required for MVP but don't hardcode `N_IO = 1`
  in a way that blocks extending it).

### 2.2 "Fully closed loop" constraint (important — this is the paper's key structural claim)

The white paper is explicit that the equilibrium feedback ONLY works if the loop is fully closed:

1. An IO neuron may only be modulated by DCN neurons that are themselves modulated by exactly
   the PKJ cells that the IO neuron's own climbing fiber contacts (no leakage to/from PKJs
   outside the IO's own target group).
2. Every PKJ that an IO neuron's CF contacts must itself help modulate that same IO (via
   PKJ→DCN→IO), i.e. no gr→PKJ synapse is under the control of an IO it doesn't help regulate.

**Implementation requirement:** connectivity must be generated so each IO "group" (1 IO + its
8–10 PKJs + its 1–2 DCN neurons) is a self-contained closed triple-negative loop. Provide a
config flag `enforce_closed_loop: bool` that, when set False, deliberately breaks constraint (1)
or (2) (e.g., shuffle which PKJs feed which DCN, or which DCN inhibits which IO) so the ablation
experiment for H1 (open- vs closed-loop drift) can be run by toggling one flag.

---

## 3. Signals and events

- **PF spikes**: independent Poisson processes per PF unit, rate configurable (e.g., 5–50 Hz),
  generated per timestep via `numpy.random.poisson` / Bernoulli-per-bin approximation at the
  chosen `dt`.
- **PKJ activity**: leaky integrator (rate or LIF) driven by sum of weighted active PF inputs
  (excitatory) minus IO's inhibitory teaching input (small, static weight — CF also causes a
  brief pause/inhibition in the PKJ in real cerebellum; keep it simple, a short fixed-duration
  inhibitory pulse on CF arrival is sufficient for the MVP) and tonic inhibition is not required
  for MVP.
- **DCN activity**: leaky integrator inhibited by summed PKJ output (PKJ is normally tonically
  active and inhibits DCN — i.e., more PKJ firing → less DCN firing → less inhibition of IO →
  more IO firing; get this sign chain right, it IS the whole point of the negative feedback loop).
- **IO activity / CF events**: IO has a baseline spontaneous firing rate (config default target:
  ~1 Hz at equilibrium in vivo per the paper) and is inhibited by DCN. Each IO spike/burst = one
  **CF event**, timestamped, and is the trigger for the plasticity rule below. (A CF event can be
  modeled as a single spike per crossing for MVP; multi-spike bursts are a stretch feature, not
  required — the paper notes burst size variability matters for the *coupling/oscillation* story,
  which is explicitly out of scope here.)

Keep neuron models as simple as possible (leaky integrate-and-fire is enough; do not implement
Hodgkin-Huxley-style ionic currents — that machinery belongs to the subthreshold-oscillation
story, which is out of scope).

---

## 4. Plasticity rule (coincidence detection LTD/LTP window) — the core mechanism

For every PF→PKJ synapse, independently:

- Maintain a short spike-time history per PF input (just needs the timestamps of recent PF
  spikes within, say, the last `LTD_WINDOW_MS + margin`).
- On every IO CF event at time `t_cf`:
  - For each PF synapse onto that IO's PKJ group: if that PF fired within
    `[t_cf - LTD_WINDOW_MS, t_cf]`, apply **LTD**: `w -= delta_minus`.
  - (Stretch/H3 variant only) if that PF fired within the null window
    `(t_cf - LTD_WINDOW_MS - NULL_WINDOW_MS, t_cf - LTD_WINDOW_MS]`, apply **no change**.
- On every PF spike that is NOT immediately followed by a CF event within the LTD (and, if
  enabled, null) window, apply **LTP**: `w += delta_plus`. (Simplest correct implementation:
  at the end of each PF spike's window without a CF event landing on it, apply LTP once per PF
  spike — implement this however is cleanest given the time-stepped loop; a clean approach is:
  process time in dt bins, and for each PF spike, schedule its LTD/LTP resolution at
  `t_spike + LTD_WINDOW_MS` (or `+ LTD_WINDOW_MS + NULL_WINDOW_MS` in the 3-window variant),
  check retrospectively whether a CF event landed in the appropriate window, then apply the
  correct update.)
- Clip weights to `[0.0, 1.0]` (paper's silent-synapse-to-max-strength convention).
- Default magnitudes (paper's worked example, 100 ms LTD window / 1 Hz equilibrium):
  `delta_plus = 0.001`, `delta_minus = 0.009` (i.e., ratio 9:1). Expose both as config and expose
  the window lengths as config so H2's sweep experiment can vary `LTD_WINDOW_MS` and observe how
  the drift-free ratio changes.

---

## 5. Experiments the MVP must be able to run (this is what "testing the hypotheses" means concretely)

1. **Baseline closed-loop run**: closed loop, default params, long enough duration (e.g.
   simulated 60–120 s) to reach equilibrium. Plot/report IO firing rate over time (should
   approach ~1 Hz) and mean PF→PKJ weight over time (should stabilize, not saturate at 0 or 1).

2. **Open-loop ablation (H1)**: same as (1) but with `enforce_closed_loop: False` or with
   DCN→IO connection weight forced to 0 (IO gets no feedback, just fires at its intrinsic
   spontaneous rate). Expect PF→PKJ weights to drift toward saturation (0 or 1) instead of
   stabilizing. Report/plot weight trajectory contrast against (1).

3. **δ+/δ− ratio sweep (H2)**: fix `LTD_WINDOW_MS`, sweep `delta_plus`/`delta_minus` ratio (or
   vice versa) and measure net drift rate of mean weight over a fixed simulated duration. Find
   the empirical ratio that yields ~zero drift and compare to the analytical prediction
   `ratio ≈ (equilibrium_interval_ms − LTD_WINDOW_MS) / LTD_WINDOW_MS`.

4. **(Stretch) Three-window variant (H3)**: enable null window, re-run experiment 3, and report
   the equilibrium δ+/δ− ratio and qualitative effect on random-walk step size (proxy for
   "forgetting" speed) — e.g., variance of weight over time at equilibrium.

Each experiment should be a runnable script/CLI entry point, not just a notebook cell, so Claude
Code (or the user) can `python -m sim.run_experiment --name closed_loop_baseline`.

---

## 6. Suggested file/module layout

```
io_equilibrium_sim/
├── README.md                  # brief usage instructions (generate at the end)
├── config.py                  # dataclass or dict-based config w/ all params + defaults above
├── sim/
│   ├── __init__.py
│   ├── poisson_input.py       # PF Poisson spike generation
│   ├── neurons.py             # PKJ / DCN / IO leaky-integrator or LIF models
│   ├── connectivity.py        # builds closed-loop (or deliberately-broken) connection tables
│   ├── plasticity.py          # coincidence-detection LTD/LTP rule, 2-window and 3-window modes
│   ├── io_coupling.py         # STUB ONLY — TODO for future electrical coupling / subthreshold
│   │                          #   oscillation work; not implemented in MVP
│   ├── simulate.py            # main time-stepped simulation loop, dt-based
│   └── analysis.py            # equilibrium rate calc, weight drift metrics, plotting helpers
├── experiments/
│   ├── run_baseline.py        # experiment 1
│   ├── run_ablation.py        # experiment 2 (H1)
│   ├── run_window_sweep.py    # experiment 3 (H2)
│   └── run_three_window.py    # experiment 4 (stretch, H3)
├── tests/
│   └── test_plasticity.py     # unit test: manually crafted spike trains → verify LTD/LTP fires
│                               #   in exactly the right windows with correct sign/magnitude
└── requirements.txt            # numpy, matplotlib (no GPU/CUDA deps — CPU only, MVP)
```

---

## 7. Non-goals / explicit simplifications (state these in the generated README too)

- No mossy fibers, granule cell layer, or Golgi cells — PF spikes are generated directly as
  Poisson processes, not derived from an upstream MF→GC circuit like full CbmSim does.
- No IO subthreshold oscillations, no IO-IO electrical coupling (explicit project-owner
  decision — stub only).
- No spatial/topographic organization beyond the group structure needed for the closed-loop
  constraint.
- Neuron models are simple leaky integrators / LIF, not conductance-based / multi-compartment.
- CF bursts are modeled as single timestamped events, not variable-spike-count bursts.

---

## 8. Definition of done for this MVP

- [ ] Config-driven, runs on CPU in seconds to low-minutes for the default scale.
- [ ] Closed-loop baseline reaches and holds a stable IO rate near the configured equilibrium
      target and stable mean PF→PKJ weight (H1 supported).
- [ ] Open-loop/ablated variant visibly drifts in weight/rate compared to baseline (H1 contrast).
- [ ] Window-length vs. δ+/δ− sweep produces a plottable curve roughly matching the analytical
      prediction in Section 5.3 (H2 supported).
- [ ] Code has the closed-loop constraint (Section 2.2) as an explicit, testable/toggleable
      piece of connectivity generation, not implicit in how populations happen to be wired.
- [ ] `io_coupling.py` exists as a documented stub for future work, not silently omitted.
- [ ] Basic unit test(s) on the plasticity window logic pass.
