# Pilot v0 data-scaling endpoint convergence diagnostic

> **Status:** Completed historical calibration stage, superseded by Pilot v0
> closure; see `docs/pilot_v0_closure.md`.

This diagnostic tests whether extending the common primary training cap from
10 to 30 complete data passes is enough for the N=1 and N=48 endpoints to
satisfy the existing convergence policy. It does not replace or overwrite the
previous capped result under `results/pilot_data_scaling_v0/`.

The 12 runs are the Cartesian product of independent training-section counts
1 and 48, true-spatial and shuffled context, and seeds 101, 202, and 303. They
reuse subset-selection seed 314159, its exact ordered section manifest, the
fixed validation observations and masks, positive 85% masking data, K=32,
the 121,024-parameter contextual model, AdamW at 3e-4 with weight decay 1e-4,
constant learning rate, and batch size 32.

The only intended training-policy change is `maximum_epochs: 30`. The absolute
step safety cap is mechanically raised from 150,000 to 400,000 because 30 full
passes at N=48 require 368,640 optimizer steps. Validation remains once per
complete pass. Early stopping still uses overall masked validation MSE, an
absolute improvement threshold of 1e-4, patience three, a minimum of two
passes, and restoration of the best accepted model state. Every selected focal
observation is presented exactly once per pass.

Only termination by the existing patience rule is classified as
`EARLY_STOP_CONVERGED`. A run reaching 30 passes is conservatively classified
as `CAP_REACHED_STILL_IMPROVING`; this stage does not introduce an independent
post-hoc flatness threshold. Other termination is `UNSTABLE_OR_OTHER`.
Pass-to-pass tail improvements remain available numerically for every gene
group.

Prepare the immutable endpoint manifest without submission:

```bash
uv run python scripts/prepare_data_scaling.py \
  --spec configs/pilot/data_scaling_endpoint_convergence_v0.yaml \
  --phase endpoint_convergence_positive_85 \
  --output results/pilot_data_scaling_endpoint_convergence_v0/manifests/endpoint.json
```

The GPU array launcher requests one RTX 6000, 8 CPUs, 64 GiB, and two hours per
task. Run it with at most eight concurrent tasks. Aggregate only after all 12
runs finish:

```bash
uv run python scripts/summarize_data_scaling_endpoint_convergence.py
```

Aggregation validates exact subset and validation-observation identity against
the earlier endpoint runs, compares pass-10 trajectories, computes final-three
and final-five tail diagnostics, calculates paired delta-space trajectories,
and produces endpoint-only plots. It fits no scaling function.
