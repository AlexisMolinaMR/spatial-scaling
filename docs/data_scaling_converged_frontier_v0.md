# Pilot v0 final converged training-data frontier

> **Status:** Historical plan, not completed as a seven-point experiment. The
> endpoint runs and 12 intermediate `N=2`/`N=4` run directories were completed;
> four `N=8` staging directories were interrupted. No complete grid or final
> aggregation was produced. See `docs/pilot_v0_closure.md`.

This stage measures validation masked-expression MSE against the number of
independent synthetic training sections at the seven fixed points 1, 2, 4, 8,
16, 32, and 48. Every section contains 8,192 focal cells, so the corresponding
available-cell counts are also reported, but section count remains the
independent-data scaling variable.

The scientific configuration is unchanged: positive synthetic corpus, 85%
high-random focal masking, K=32 exact same-section Euclidean context, true
spatial and validated shuffled arms, the 121,024-parameter contextual model,
batch size 32, and seeds 101/202/303. Validation observations and masks remain
fixed. Subsets use seed 314159 and exact prefixes of the previously validated
section ordering.

The frozen convergence policy is AdamW at base LR 3e-4 and weight decay 1e-4,
with validation-plateau reductions by factor 0.3 after three insufficient
pass-end checks, a 1e-4 absolute improvement threshold, LR floor 1e-5, and
early stopping after eight insufficient checks. Validation occurs once per
complete deterministic pass. The maximum 90-pass and 1,200,000-step limits are
safety ceilings. Every selected focal observation is consumed once per pass.

The planned logical grid contains 42 runs. Twelve completed converged endpoint
runs were available for reuse:
N=1 and N=48 spatial runs from the 90-pass diagnostic and their matched
shuffled runs from the plateau-decay arm of the 60-pass diagnostic. Reuse is
accepted only after exact data, model, optimizer, scheduler, evaluation-hash,
subset, coverage, and convergence validation. The older shuffled ceiling is
allowed to differ only because those runs satisfied the same early-stopping
rule before that ceiling. Of the remaining 30 planned N=2/4/8/16/32 runs, all
12 N=2/N=4 tasks produced complete-looking run directories before closure and
four N=8 tasks left interrupted staging directories. The partial output was
not aggregated or interpreted. The rest of the grid was not pursued.

Prepare the immutable new-run manifest without submission:

```bash
uv run python scripts/prepare_data_scaling.py \
  --spec configs/pilot/data_scaling_converged_frontier_v0.yaml \
  --phase final_converged_positive_85 \
  --n-sections 2 --n-sections 4 --n-sections 8 \
  --n-sections 16 --n-sections 32 \
  --output results/pilot_data_scaling_converged_frontier_v0/manifests/new_runs.json
```

The preserved GPU array defines exactly those 30 unexecuted new runs. The
preserved final aggregator would validate all 42 logical runs before computing
paired seed-level spatial advantages,
mean and sample-SD summaries, observed adjacent improvements, convergence and
compute tables, capped-versus-converged comparisons, and plots. It fits no
scaling function or exponent.
