# Spatial Scaling Pilot v0 closure

## 1. Purpose of Pilot v0

Pilot v0 was designed as a controlled measurement instrument for separating
and measuring training-data scale, spatial-context scale, spatial organization
versus context quantity, and computational scaling. Its synthetic phase was a
calibration system with known falsification controls, not a biologically
realistic tissue simulator or a model of a particular spatial technology.

The observation and model token were synthetic cells. Synthetic sections were
the independent experimental and split units. The learning target was masked
continuous expression.

## 2. Final project status

> Pilot v0 was closed following a project-level research pivot.

> Closure occurred after the principal synthetic calibration and
> spatial-control methodology had been validated, but before the originally
> planned final converged N_train frontier, formal scaling-law fitting, or
> real-data extension.

This was an intentional project-level decision, not a scientific failure of
Pilot v0. The code and specifications are preserved to reproduce the completed
calibrations and to record the unexecuted final design.

At closure, ignored output storage contained 12 complete-looking intermediate
run directories at `N=2` and `N=4`, plus four interrupted `N=8` staging
directories. These partial artifacts were not aggregated or interpreted as a
frontier. In this record, the final seven-point sweep being "not executed"
means that the predeclared experiment was not completed and validated as a
seven-point scientific result; it does not mean that no intermediate task had
ever started.

## 3. Validated synthetic generator

The validated positive and `lambda_s=0` null corpora each contain 64 sections:
48 training, 8 validation, and 8 test sections. Every section contains 8,192
cells, for 524,288 cells per corpus. Generation and section-level splits are
deterministic, and sections are generated independently.

Representative full-corpus short-range expression correlations were:

| World | Spatial genes | Mixed genes | Intrinsic genes |
| --- | ---: | ---: | ---: |
| Positive | ~0.899 | ~0.425 | ~0.006 |
| Null | ~-0.001 | ~0.006 | ~0.006 |

Cross-section absolute correlation was approximately 0.0022. The configured
spatial correlation length was 200 µm. Direct generated-field analysis yielded
approximately 202.6 µm, whereas the ordinary finite-section QC estimator
yielded approximately 177.1 µm. The latter discrepancy was diagnosed as bias
from finite-section centering and normalization in the estimator, not bias in
the generator.

## 4. SSL objective validation

The focal-only masked molecular modeling objective used masked expression plus
a binary visibility vector. Masking was deterministic per cell, and MSE was
computed only for masked targets. Metrics were reported overall and separately
for intrinsic, mixed, and spatial genes. The focal MLP had 115,200 parameters.

Representative mask calibration was:

| Masked fraction | Validation MSE |
| ---: | ---: |
| 15% | ~0.0732 |
| 30% | ~0.1067 |
| 50% | ~0.1734 |
| 70% | ~0.2956 |
| 85% | ~0.4899 |

The frozen conventional baseline used 30% random masking. The spatially
demanding regime used 85% high-random masking.

## 5. Spatial-vs-shuffled falsification

The validated context was exact within-section Euclidean KNN with `K=256` for
the initial falsification. The focal cell was excluded, all context came from
the same section, and relative coordinates were expressed in µm.

The shuffled control used a seeded bijective wrong-anchor derangement. It drew
locally coherent molecular cells around the wrong anchor, excluded the focal
cell and all true-KNN identities, and retained the true relative-coordinate
slots. Architecture, masks, initialization, optimizer, and training steps were
matched to the spatial arm.

The contextual model had 121,024 parameters, 64-dimensional embeddings, two
transformer layers, four attention heads, FFN width 128, and a
relative-coordinate MLP.

For

```text
delta_space = L_shuffled - L_spatial,
```

the decisive positive-world 85% result was approximately:

| Gene group | delta_space |
| --- | ---: |
| Intrinsic | -0.0011 |
| Mixed | +0.2245 |
| Spatial | +0.4574 |

The spatial-gene result was positive in 3/3 seeds. Null-world spatial
`delta_space` was approximately zero. This validated the instrument's ability
to detect correctly organized spatial information while rejecting a
non-spatial null.

## 6. Context-size scaling

The completed sweep used `K = 32, 64, 128, 256, 512, 1024`.

> Spatial advantage was already essentially saturated by K=32.

Representative spatial-gene `delta_space` values were:

| K | Spatial-gene delta_space |
| ---: | ---: |
| 32 | ~0.4599 |
| 64 | ~0.4534 |
| 128 | ~0.4466 |
| 256 | ~0.4574 |
| 512 | ~0.4569 |
| 1024 | ~0.4556 |

Intrinsic effects remained approximately zero, as did the null control. K
alone did not recover the planted physical scale because even a small number
of nearest cells can be extremely local and redundant.

## 7. Physical-distance calibration

At low K, `K=1` retained approximately 98.6% of the `K=32` spatial-gene
advantage. Its nearest-cell median distance was approximately 10.4 µm.

The fixed-`K=32` distance-exclusion experiment used
`d_min = 0, 50, 100, 150, 200, 300, 400, 600 µm`.

| d_min | Spatial-gene delta_space |
| ---: | ---: |
| 0 µm | ~0.4599 |
| 50 µm | ~0.4510 |
| 100 µm | ~0.4134 |
| 150 µm | ~0.3329 |
| 200 µm | ~0.2267 |
| 300 µm | ~0.0675 |
| 400 µm | ~0.0090 |
| 600 µm | ~-0.0065 |

Representative realized median selected-cell distances were approximately
50.6 µm at the 0 µm threshold, 207.0 µm at 200 µm, 305.1 µm at 300 µm,
404.2 µm at 400 µm, and 603.4 µm at 600 µm.

> Cell-count context and physical context geometry are distinct scaling
> variables.

The planted approximately 200 µm RBF scale was resolved much more clearly by
manipulating physical context distance than by increasing K. Prediction
advantage is not equated with raw RBF correlation.

## 8. Computational context scaling

Representative dense-context costs were approximately 31 ms/step at `K=32`,
34 ms/step at `K=256`, 54 ms/step at `K=1024`, and 87–94 ms/step in the
`K=2048` probe. Peak reserved GPU memory for `K=2048` was approximately
1.15 GiB.

> Dense attention remained practical through the tested K=2048 range for
> Pilot v0, although computational cost increased materially.

`K=4096` was not pursued. With 8,192-cell sections, the validated shuffled
control's focal and true-context exclusion rule requires at least
`2*K + 1 = 8,193` cells, so a matched control was structurally infeasible and
unnecessary for this calibration.

## 9. Training-data scaling work

Nested section-level scaling was implemented for
`N_train = 1, 2, 4, 8, 16, 32, 48`, using subset seed `314159`. Each subset was
an ordered prefix of one fixed training-section permutation. Validation and
test sections remained fixed.

The completed 10-pass result is a **capped learning curve**, not the final
converged frontier. Representative overall mean validation losses were:

| N sections | Spatial | Shuffled |
| ---: | ---: | ---: |
| 1 | ~0.3037 | ~0.6102 |
| 2 | ~0.2735 | ~0.5533 |
| 4 | ~0.2639 | ~0.5346 |
| 8 | ~0.2551 | ~0.5229 |
| 16 | ~0.2453 | ~0.5134 |
| 32 | ~0.2347 | ~0.4936 |
| 48 | ~0.2296 | ~0.4806 |

Both arms showed a strong, stable data-scaling signal. The fixed 2,000-step
control under-exposed larger datasets. Fixed-compute scaling and
converged-data scaling were therefore correctly treated as distinct
experimental objects.

## 10. Optimization/convergence calibration

### 10-pass cap

Ten data passes were insufficient to establish a converged frontier.

### 30-pass endpoint test

Thirty passes remained insufficient for the spatial endpoint models.

### Constant LR versus plateau decay

Constant-LR early stopping was shown to stop at temporary optimization
plateaus. Plateau-triggered learning-rate decay improved every endpoint arm.

### Frozen convergence policy

The final calibrated policy used:

* AdamW;
* base learning rate `3e-4`;
* plateau-triggered reductions `3e-4 -> 9e-5 -> 2.7e-5 -> 1e-5`;
* unchanged early-stopping semantics;
* batch size 32;
* a 90-pass safety cap.

All six spatial endpoint models at `N=1` and `N=48` reached the learning-rate
floor, received 10–30 validation checks after the final reduction, early
stopped before pass 90, and reproduced their previous histories exactly.

Terminal endpoint spatial advantages were approximately:

| N | Overall | Intrinsic | Mixed | Spatial |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3205 | 0.0185 | 0.3134 | 0.6271 |
| 48 | 0.2379 | 0.0622 | 0.2366 | 0.4143 |

> The optimization protocol was successfully calibrated, but the final
> converged seven-point N_train sweep was not executed because the project
> pivot occurred at this point.

The partial intermediate artifacts noted in Section 2 do not change this
status: there is no complete seven-point grid and no final frontier summary.

## 11. Unresolved intrinsic-gene effect

Intrinsic genes were constructed with zero planted spatial loading. Despite
this, prolonged optimization produced a positive intrinsic `delta_space`,
especially at larger N: approximately 0.0185 at `N=1` and 0.0622 at `N=48`.

Undertraining did not explain the effect, the null control remained
approximately zero, and no direct mechanical leakage was identified.
Shared-representation or multi-task coupling is one possible hypothesis, but
no mechanistic ablation was run and no hypothesis is established. This remains
an unresolved modeling question.

## 12. What was not completed

The following experiments and analyses were not pursued after the pivot; they
did not fail:

* completion, validation, and aggregation of the final converged
  `N_train = 1/2/4/8/16/32/48` sweep;
* formal scaling-law fitting;
* scaling exponent estimation;
* model-size scaling;
* compute-optimal scaling;
* donor, sample, or tissue scaling;
* gene-vocabulary or data-quality scaling;
* real-data scaling;
* validation on real spatial transcriptomics;
* mechanistic investigation of intrinsic `delta_space`.

## 13. Final scientific conclusions

Pilot v0 established that:

1. the synthetic instrument contains controlled and falsifiable spatial
   structure;
2. the SSL objective is stable and leakage-safe;
3. correct spatial organization provides large predictive information not
   available to an architecture-matched shuffled control;
4. the effect disappears in the null world;
5. context cell count and physical spatial distance are not interchangeable;
6. physical-distance exclusion qualitatively recovers the planted spatial
   scale;
7. training-data scale produces a strong empirical learning signal;
8. optimization convergence materially changes apparent scaling curves;
9. a defensible convergence policy was calibrated;
10. the full converged data frontier was not run because of the project pivot.

Pilot v0 does not establish a universal scaling law or scaling laws in real
biology. It successfully validated its synthetic spatial measurement machinery
and calibrated its training protocol, then was intentionally closed before the
final scaling-law program when the research direction pivoted.
