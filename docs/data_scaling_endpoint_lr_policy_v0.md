# Pilot v0 endpoint learning-rate policy diagnostic

> **Status:** Completed historical calibration stage, superseded by Pilot v0
> closure; see `docs/pilot_v0_closure.md`.

This diagnostic compares two globally fixed convergence policies at only N=1
and N=48 independent synthetic training sections. It preserves the positive
corpus, 85% high-random focal masking, K=32 same-section context, spatial and
shuffled controls, model seeds 101/202/303, subset seed 314159, validation
observations/masks, batch size 32, AdamW, base LR 3e-4, weight decay 1e-4, and
the 121,024-parameter contextual model. It does not run intermediate data
scales or fit a scaling law.

`prolonged_constant_lr` retains the established overall masked-MSE stopping
rule: improvement threshold 1e-4, patience three validation checks, minimum two
complete passes, constant LR 3e-4, and a 60-pass maximum.

`validation_plateau_lr_decay` uses one shared policy for every endpoint:

* improvement threshold: 1e-4 overall validation masked MSE;
* LR-reduction patience: three insufficient validation checks;
* reduction factor: 0.3;
* minimum LR: 1e-5;
* early-stopping patience: eight insufficient validation checks;
* maximum: 60 complete passes.

On an accepted improvement, both counters reset. On three insufficient checks,
LR is reduced and the early-stop opportunity window resets. The reduction
event cannot stop training on the same validation. Once LR is at its floor,
eight insufficient checks after the last improvement are required to stop.
Validation remains once per complete deterministic data pass, and every active
observation appears exactly once in every pass.

The absolute safety cap is 800,000 optimizer steps: 60 complete N=48 passes
require 737,280 steps. Every history records the LR used for the completed pass,
the LR after the validation decision, reduction events, and early-stop events.

The 24-run grid is two N values by two context conditions by three seeds by two
policies. Results use the distinct namespace
`results/pilot_data_scaling_endpoint_lr_policy_v0/` and link to the prior
30-pass endpoint diagnostic for exact overlapping-history reproduction.

Prepare, but do not submit, the immutable manifest with:

```bash
uv run python scripts/prepare_data_scaling.py \
  --spec configs/pilot/data_scaling_endpoint_lr_policy_v0.yaml \
  --output results/pilot_data_scaling_endpoint_lr_policy_v0/manifests/endpoint_lr_policy.json
```

The array launcher requests one RTX 6000, 8 CPUs, 64 GiB, and three hours per
task, with at most eight concurrent tasks. Aggregation validates identities,
subsets, observations, masks, historical reproduction, and pre-reduction
reproduction before producing policy, delta-space, tail, LR-event, and compute
tables and plots.
