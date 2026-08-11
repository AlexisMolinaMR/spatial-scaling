# Pilot v0 spatial-context scaling harness

## Scientific question

This stage measures the seed-paired masked-expression advantage of correctly
organized context over the validated shuffled-context control as exact
within-section KNN size increases:

```text
delta_space(K) = L_shuffled(K) - L_spatial(K).
```

The observation and model token are synthetic cells. The independent split
unit is the synthetic section. The target is raw focal-cell expression at
deterministically masked genes. The primary scale points are K = 32, 64, 128,
256, 512, and 1024. Evaluation remains on the fixed section-level validation
split and fixed 8,192-cell observation subset. No power law is fitted.

All definitions in `docs/spatial_falsification_v0.md` remain frozen, including
the true-spatial context, seeded same-section wrong-anchor derangement,
exclusion of focal and true-KNN molecular identities, retention of true
relative-coordinate slots, focal masking, gene-class metrics, model, corpus,
and section splits.

## Controlled quantities

The primary positive 85% masking sweep uses matched seeds 101, 202, and 303 for
the spatial and shuffled arms. AdamW, learning rate 3e-4, weight decay 1e-4,
constant learning rate, 2,000 steps, batch size 32, 8,192 fixed validation
observations, model initialization, masks, and evaluation masks remain matched.
The contextual architecture remains 64-dimensional with two transformer
layers, four attention heads, FFN width 128, and a 200 µm coordinate scale.
K changes token count and computation but not model parameters.

The minimum null control uses K = 32, 256, and 1024 under 85% masking. The
secondary positive 30% masking control uses K = 64, 256, and 1024. Both use the
same matched conditions and three seeds.

## Physical receptive field

For every fixed evaluation target, the harness measures the distance in saved
physical micrometres to its true Kth nearest neighbor. It reports the median,
5th, 25th, 75th, and 95th percentiles. This geometry-only measurement is
independent of which molecular identities populate shuffled context slots. The
known generator RBF correlation scale is 200 µm, but no equality between that
parameter and predictive saturation is assumed.

## Experiment manifests and outputs

`configs/pilot/context_scaling_v0.yaml` is the single sweep specification.
`scripts/prepare_context_scaling.py` deterministically expands a selected phase
to an immutable JSON manifest. Every array index records its corpus, masking
regime, K, condition, seed, training budget, full resolved model configuration,
output path, experiment ID, and configuration SHA-256 identity.

`slurm/train_context_scaling.sbatch` runs one recorded manifest entry. A run is
written to a unique staging directory and atomically published only after all
metrics and provenance are complete. Existing completed outputs are never
overwritten. `scripts/summarize_context_scaling.py` validates matched settings,
parameter counts, masks, observations, steps, batch sizes, and effective
training examples before calculating seed-level deltas.

## Computational measurements

Each run records total elapsed time, optimization-only elapsed time, mean and
median synchronized training-step time, examples and context tokens per second,
and peak allocated and reserved GPU memory. The primary batch size and training
budget remain fixed across K.

## Stop gates and feasibility boundary

The K=256 tasks run first. `scripts/check_context_scaling_anchor.py` compares
their seed-level deltas with the validated falsification artifacts and fails if
model capacity changes, spatial benefit is not positive in 3/3 seeds, or
per-seed deviations exceed predeclared numerical tolerances.

The K=2048 resource probe is a single matched seed-101 pair with batch size 32,
50 optimization steps, and 64 fixed evaluation observations. It is a compute
probe, not part of the scientific receptive-field curve. It is attempted only
after successful K=1024 runs. A matched K=4096 control is structurally
impossible in the current 8,192-cell sections: the validated exclusion rule
requires at least `2*K + 1 = 8,193` cells. The control will not be weakened to
force that point.

## Follow-up spatial-scale calibration

The nearly flat K = 32...1024 curve motivated a frozen-design diagnostic at
K < 32 and at fixed K = 32 with explicit minimum-distance exclusion. Its exact
selection, matched shuffled extension, common eligibility policy, realized
distance measurements, and experiment grid are documented in
`docs/spatial_scale_calibration_v0.md`. This calibration changes physical
context geometry; it does not begin training-data scaling.
