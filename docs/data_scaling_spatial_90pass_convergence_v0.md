# Pilot v0 90-pass spatial endpoint convergence diagnostic

> **Status:** Completed historical calibration stage. Its frozen policy was not
> used for the planned full seven-point frontier following the project pivot;
> see `docs/pilot_v0_closure.md`.

This diagnostic asks whether the unchanged validation-plateau learning-rate
policy reaches a stable frontier when its maximum safety cap is extended from
60 to 90 complete training-data passes. It runs only true-spatial context at
N=1 and N=48 independent synthetic sections for seeds 101, 202, and 303.
Intermediate N values, shuffled arms, null controls, fixed-compute controls,
and scaling-law fits are excluded.

The pass-60 outputs are not scientifically resumable: they contain resolved
configuration, provenance, validation history, metrics, and plots, but no
model, optimizer, scheduler, early-stop, sampler, or RNG state. All six runs
therefore restart from pass zero. Aggregation requires exact reproduction of
every pass from zero through 60 before interpreting the extension.

The scientific configuration remains positive synthetic data, 85% high-random
focal masking, K=32 exact same-section Euclidean context, the 121,024-parameter
contextual model, AdamW at base LR 3e-4 and weight decay 1e-4, batch size 32,
subset seed 314159, and model seeds 101/202/303. The plateau policy remains:

* improvement threshold 1e-4 on overall validation masked MSE;
* LR-reduction patience three validation checks;
* reduction factor 0.3;
* minimum LR 1e-5;
* early-stop patience eight validation checks;
* validation once after each complete deterministic data pass.

Only `maximum_epochs` changes from 60 to 90. The absolute safety cap is raised
mechanically from 800,000 to 1,200,000 steps because 90 N=48 passes require
1,105,920 optimizer steps.

The previously converged plateau-decay shuffled endpoints are reused only for
paired spatial-advantage aggregation. For passes present in both histories,
delta uses matched pass-specific losses. After a shuffled history ends, its
restored converged endpoint is held fixed while the spatial trajectory
continues.

Prepare the immutable six-run manifest without submission:

```bash
uv run python scripts/prepare_data_scaling.py \
  --spec configs/pilot/data_scaling_spatial_90pass_convergence_v0.yaml \
  --output results/pilot_data_scaling_spatial_90pass_convergence_v0/manifests/spatial_90pass.json
```

The GPU launcher requests one RTX 6000, 8 CPUs, 64 GiB, and three hours per
task. Aggregation validates the exact grid, scientific invariants, full
coverage, 0–60 reproduction, scheduler trajectories, post-final-reduction
tails, and pairing to the historical shuffled endpoints before writing plots
and tables. It fits no scaling function.
