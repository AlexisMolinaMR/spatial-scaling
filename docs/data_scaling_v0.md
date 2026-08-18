# Pilot v0 training-data scaling

> **Status:** Archived after completion of the capped learning curve and
> convergence-method calibration. The final converged seven-point frontier was
> not completed or aggregated following the project pivot; see
> `docs/pilot_v0_closure.md`.

## Scientific scope

This stage measures masked molecular prediction as the number of independent
synthetic training sections increases. The observation and model token are a
synthetic cell; the independent experimental, subset-selection, and split unit
is a synthetic section. The scaling axis is `N_train_sections`, with observed
points 1, 2, 4, 8, 16, 32, and 48. Because every Pilot v0 section contains
8,192 cells, these points correspond deterministically to 8,192 through
393,216 available focal cells. Section count and cell count are both reported
and are not treated as interchangeable concepts outside this equal-size
synthetic corpus.

The validation distribution is the unchanged eight-section validation split.
Exactly 8,192 validation focal observations are selected with observation seed
991, and their 85% high-random masks use the fixed evaluation mask seed 992.
The test split remains untouched. The input modality is continuous synthetic
expression, not a particular spatial technology. Relevant confounding is that
primary frontier compute grows with the amount of selected training data; this
is measured explicitly and checked against a separate fixed-compute protocol.

## Controlled model and context

Every contextual run uses K=32 exact within-section Euclidean neighbors, fully
observed context expression, masked focal expression, and the validated true
spatial or matched shuffled-context construction. Context never crosses a
section, and training datasets expose only the active section subset. The
contextual architecture remains the 121,024-parameter shared
expression-plus-visibility encoder and spatial transformer: 64-dimensional
embeddings, two transformer layers, four heads, FFN width 128, the existing
relative-coordinate MLP, and a 200 µm coordinate scale. AdamW uses learning
rate 3e-4, weight decay 1e-4, contextual batch size 32, and a constant learning
rate schedule.

## Nested section subsets

`configs/pilot/data_scaling_v0.yaml` fixes subset-selection seed 314159. A
stable BLAKE2b-addressed NumPy permutation of the corpus's 48 fixed training
section IDs is constructed once. Every scale is an ordered prefix of this same
permutation. Model seeds 101, 202, and 303 and both contextual arms reuse the
same membership. The manifest records the complete ordering and active prefix
for every task. Cells are not subsampled within selected sections in the
primary experiment.

## Converged-data frontier policy

The primary `converged_data_frontier` protocol uses deterministic shuffled
passes. Within each pass, section order and cell order within each section are
seeded and shuffled, and every active focal observation appears exactly once.
This construction keeps section exposure balanced and makes the spatial and
shuffled arms' focal order identical for a matched model seed.

The common stopping policy is:

* validation once after every complete pass;
* masked validation MSE overall as the stopping metric;
* early stopping prohibited before two complete passes;
* absolute improvement threshold 1e-4;
* patience of three pass-end validations;
* maximum 10 complete passes;
* absolute safety cap 150,000 optimizer steps;
* restore the best accepted validation state.

The step cap exceeds the 122,880 steps required for ten complete passes at
N=48 and therefore should not truncate the planned policy. Any cap that prevents
minimum exposure fails the run. Each result records actual steps, examples,
effective passes, unique focal observations, per-section exposure, stopping
reason, runtime, throughput, and peak GPU memory.

## Fixed-compute sensitivity

The secondary `fixed_compute_2000_steps` protocol is run at 1, 8, and 48
sections. It uses exactly 2,000 steps and the historical seeded random-section,
random-cell sampler with replacement. It is not used as the primary data
frontier. Its measured unique-observation coverage is reported so that the
larger available pool is not mistaken for data actually seen during training.

## Experiment matrix

The required contextual runs are:

* positive converged frontier: 7 scales × 2 conditions × 3 seeds = 42;
* null converged controls: 3 scales × 2 conditions × 3 seeds = 18;
* positive fixed-compute sensitivity: 3 scales × 2 conditions × 3 seeds = 18.

The established focal-only model is also run as a converged reference at all
seven scales and three seeds (21 runs). Its architecture, parameter count, and
batch size differ from the contextual model, so it is not interpreted as a
pure measure of spatial information. Sparse 30% masking is omitted in this
stage so it cannot delay the required 85% grid.

Paired spatial advantage is calculated within each N and seed before mean,
sample standard deviation, and sign consistency are summarized:

```text
delta_space(N) = L_shuffled(N) - L_spatial(N).
```

No scaling-law functional form, exponent, or extrapolation is fitted.

## Reproducible execution

Prepare an immutable manifest without submission:

```bash
uv run python scripts/prepare_data_scaling.py \
  --phase primary_positive_85 \
  --output results/pilot_data_scaling_v0/manifests/primary.json
```

Run one manifest through the GPU array launcher:

```bash
sbatch --array=0-41%8 slurm/train_data_scaling.sbatch \
  results/pilot_data_scaling_v0/manifests/primary.json
```

The production launcher requests one RTX 6000, 8 CPUs, 64 GiB, and 3 hours per
task. The limit leaves headroom over the measured K=32 step time while remaining
short enough for scheduler backfill; the 150,000-step policy cap is stricter.

After all required non-smoke runs finish, aggregate without fitting a curve:

```bash
uv run python scripts/summarize_data_scaling.py
```
