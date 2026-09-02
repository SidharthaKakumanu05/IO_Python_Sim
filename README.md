# Project_1Hz

A GPU-accelerated (CuPy) spiking simulation of a minimal olivo-cerebellar loop —
four small neuron populations wired into one feedback loop, plus a learning rule.
This doc is meant to let someone with zero context on this repo understand what it
does and find their way around the code.

## The big picture

Four populations, wired one-way into a loop, with one learned connection:

```
PF ──excitatory, plastic──▶ PKJ ──inhibitory──▶ DCN ──inhibitory──▶ IO ──inhibitory──▶ PKJ
                                                                      IO ◀─gap junctions─▶ IO
```

- **PF** (parallel fibers, 4096 cells) — random background input, fires at a fixed rate. Nothing upstream of it.
- **PKJ** (Purkinje cells, 128) — a standard leaky integrate-and-fire (LIF) neuron.
- **DCN** (deep cerebellar nuclei, 8) — same LIF neuron, but with a constant bias current so it fires steadily unless suppressed.
- **IO** (inferior olive, 16) — a pacemaker-style neuron with a threshold that jumps up after each spike and decays back down over time; also directly electrically coupled to other IO cells in pairs (gap junctions), independent of synapses.

IO firing is the "teaching signal": every time an IO cell fires, it nudges the
weights of the PF→PKJ synapses feeding the PKJ cells it's paired with. Everything
else in the loop (PKJ→DCN, DCN→IO, IO→PKJ) is inhibitory and fixed-weight — only
PF→PKJ learns.

## Walking through the code

Read the files in roughly this order to build up the full picture:

1. **`config.py`** — every tunable number lives here: population sizes, timestep,
   simulation length, all neuron/synapse parameters, plasticity settings. Start
   here to see what's configurable before looking at any logic.

2. **`connectivity.py`** — builds the random wiring once at startup: which PF
   cells connect to which PKJ cells, how PKJ cells group into DCN/IO targets,
   and which IO cells are paired for gap junctions. Returns a dict of
   `pre_idx`/`post_idx` index arrays that every other file consumes.

3. **`neurons.py`** — the neuron models themselves.
   - `NeuronState` + `lif_step`: the plain LIF neuron used by PKJ and DCN.
   - `IOState` + `io_step`: the IO pacemaker neuron (decaying threshold, gap
     current, noise-driven spontaneous firing).

4. **`coupling.py`** — the IO↔IO gap junction current. Given a pair of coupled
   neurons, current flows between them proportional to their voltage
   difference — this is separate from synapses, no spikes involved.

5. **`synapses.py`** — the `Synapse` class used for every synaptic connection
   (PF→PKJ, IO→PKJ, PKJ→DCN, DCN→IO). On a presynaptic spike, conductance jumps
   and then decays exponentially; postsynaptic current is
   `conductance × (reversal_potential − postsynaptic_voltage)`, which is what
   makes a synapse excitatory or inhibitory. Each connection type also has a
   fixed transmission delay.

6. **`plasticity.py`** — the learning rule for PF→PKJ weights. Every PF fiber's
   recent spike history is tracked in a ring buffer split into two windows: the
   last 10ms counts toward depression (LTD), the 90ms before that counts toward
   potentiation (LTP), with LTD weighted 9x stronger per spike. Weights only
   update at the instant an IO cell fires. Because the window split (1:9) and
   strength ratio (9:1) are reciprocal, random uncorrelated PF activity produces
   zero net weight drift — only PF activity correlated with IO firing actually
   moves the weights.

7. **`inputs.py`** — small helpers: Poisson spike generation for PF, and random
   draws for per-synapse conductance values.

8. **`recorder.py`** — bins spikes from every population over the course of the
   run and saves everything (plus weight trajectories) to a single `.npz` file
   at the end.

9. **`simulate.py`** — ties everything above together. `run()` builds the
   populations, connectivity, and synapses from `config.py`, then loops one
   timestep at a time: draw PF spikes → PF+IO current into PKJ → step PKJ →
   PKJ current into DCN → step DCN → DCN current + gap junctions + noise into
   IO → step IO → update PF→PKJ weights from this step's IO spikes → record.
   This is the file to read to see the actual step-by-step control flow.

10. **`analysis.py`** — loads the saved `.npz` and plots spike rasters per
    population and the PF→PKJ weight trajectory over time.

11. **`main.py`** — the entry point. Checks a CUDA GPU is available, then calls
    `simulate.run()`.

## Running it

```bash
python3 main.py       # runs the simulation, saves cbm_py_output.npz
python3 analysis.py   # loads that file, produces raster/weight plots
```

or `./run_simulation.sh` to do both. `./clean_outputs.sh` deletes generated
output files (`cbm_py_output.npz`, plot directories) to reset to a clean state.

## Reading the output

- **Rasters** (one per population) — each dot is a spike; a healthy run shows
  PF firing at its fixed background rate, PKJ firing at a moderate rate shaped
  by PF input and IO inhibition, DCN firing steadily except when suppressed by
  PKJ, and IO firing sparsely (~1 Hz) and roughly evenly across cells.
- **Weight trajectory** — the mean and spread of PF→PKJ weights over time.
  A stable run keeps the mean roughly flat (no runaway drift) while individual
  weights still spread out from learning.
