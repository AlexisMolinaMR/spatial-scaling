# Spatial Scaling Pilot v0 synthetic corpus

The Pilot v0 generator is a calibration instrument for spatial experimental
machinery. It is not a biological tissue simulator. Its observation unit is a
generic synthetic cell, its independent experimental and split unit is a
section, and its molecular observations are continuous expression values rather
than sequencing counts from a particular spatial technology.

## Generator

For cell coordinates `r_i = (x_i, y_i)` in local micrometres, the generator
samples independent intrinsic states `c_i ~ N(0, I)` and an RBF-correlated
spatial state `s_i`. Expression and its saved ground-truth decomposition are

```text
intrinsic_i = c_i W_c
spatial_i   = lambda_s s_i W_s
noise_i     ~ N(0, noise_std^2 I)
expression_i = intrinsic_i + spatial_i + noise_i.
```

Each spatial latent dimension uses Random Fourier Features:

```text
s_q(r) = sqrt(2 / M) sum_m a_qm cos(omega_qm^T r + b_qm),
omega_qm ~ N(0, correlation_length_um^-2 I),
a_qm ~ N(0, 1),  b_qm ~ Uniform(0, 2 pi).
```

Thus its ensemble covariance approximates
`exp(-distance^2 / (2 correlation_length_um^2))` without an all-cell covariance
matrix. `correlation_length_um` is the distance where the ideal RBF correlation
equals `exp(-1/2)`, about 0.607. Runtime is approximately linear in cell count
for fixed latent and Fourier-feature dimensions.

`W_c` and `W_s` are globally shared random unit-norm latent projections scaled
by class coefficients:

| Gene class | Intrinsic coefficient | Spatial coefficient |
| --- | ---: | ---: |
| intrinsic | 1 | 0 |
| mixed | `1 / sqrt(2)` | `1 / sqrt(2)` |
| spatial | 0 | 1 |

The unit-norm projections make each unscaled source have approximately unit
variance under standard-normal latent states. The clean zero loadings are
intentional falsification controls.

`lambda_s` multiplies only the spatial expression contribution. The positive
config sets it to 1 and the null config sets it to 0. Coordinates, intrinsic
states, spatial fields, noise, vocabulary, and global loadings are otherwise
generated identically for a shared master seed.

## Sections, splits, and reproducibility

Coordinates, intrinsic states, spatial-field frequencies/phases/coefficients,
and observation noise are sampled independently for every section. Vocabulary
and loadings are shared. The default IDs are assigned deterministically to 48
train, 8 validation, and 8 test sections; a cell always inherits its section's
split.

Random streams are addressed by conceptual components such as
`(master_seed, section_id, "coordinates")`. A BLAKE2b digest provides a stable
integer seed; Python's process-randomized `hash()` and a globally consumed RNG
are not used. Consequently, standalone generation of a section equals its
generation inside a corpus, and section order does not affect values.

## Output schema

The corpus is section-sharded to avoid a monolithic serialization:

```text
OUTPUT/
├── metadata.json
├── genes.csv
├── gene_loadings.npz
├── sections/
│   ├── section_0000.npz
│   └── ...
└── qc/
    ├── metrics.json
    ├── spatial_latent_correlation.png
    ├── expression_correlation_by_gene_class.png
    ├── variance_decomposition.png
    └── section_independence.png
```

Each section shard contains `cell_id`, `section_id`, `split`, `x_um`, `y_um`,
`expression`, `intrinsic_latent`, `spatial_latent`, `intrinsic_expression`,
`spatial_expression`, and `noise_expression`. Numeric matrices use float32.
Gene CSV metadata records the class and scalar coefficients; the NPZ records
the complete intrinsic and spatial loading matrices. Corpus metadata includes
the resolved configuration, split IDs, generator version, master seed,
scientific parameters, geometry, dimensions, counts, and Git commit when
available.

An output directory must be absent or empty. Existing corpus contents are not
silently overwritten.

## Quantitative QC

QC samples reproducible random cell pairs, bins their physical distances, and
computes mean products after feature-wise centring and variance scaling. It
reports spatial-latent and expression correlation curves, pair counts, the
ideal RBF curve, and a fitted length scale from log correlation versus squared
distance. The finite rectangular section and centring cause modest deviation
from the infinite-domain RBF curve, so the estimate is a calibration statistic,
not an exact parameter identity.

Expression correlation is reported separately for intrinsic, mixed, and
spatial genes. Variances of the saved intrinsic, spatial, and noise components
are reported overall and by class. Section independence is checked using the
off-diagonal correlations of matched-index spatial states across independently
generated sections. In the null condition the latent spatial field remains
available for falsification, while its contribution to expression is exactly
zero.

## Commands

Prepare the environment once from the repository root:

```bash
uv sync --locked
```

Generate small positive and null corpora locally for smoke testing:

```bash
uv run python scripts/generate_synthetic.py \
  --config configs/pilot/synthetic_spatial.yaml \
  --output outputs/pilot/synthetic_spatial_small \
  --override num_sections=4 \
  --override cells_per_section=512 \
  --override splits.train=2 \
  --override splits.validation=1 \
  --override splits.test=1 \
  --override spatial_field.num_fourier_features=64 \
  --override qc.pairs_per_section=20000 \
  --override qc.max_sections=4

uv run python scripts/generate_synthetic.py \
  --config configs/pilot/synthetic_null.yaml \
  --output outputs/pilot/synthetic_null_small \
  --override num_sections=4 \
  --override cells_per_section=512 \
  --override splits.train=2 \
  --override splits.validation=1 \
  --override splits.test=1 \
  --override spatial_field.num_fourier_features=64 \
  --override qc.pairs_per_section=20000 \
  --override qc.max_sections=4
```

The default 64-section corpora are substantial CPU preprocessing and must run
through SLURM. The resolved request is CPU partition, 8 CPUs, 32 GiB memory,
and 12 hours. Create `logs/` before submission:

```bash
mkdir -p logs
sbatch slurm/generate_synthetic_pilot.sbatch \
  configs/pilot/synthetic_spatial.yaml \
  outputs/pilot/synthetic_spatial

sbatch slurm/generate_synthetic_pilot.sbatch \
  configs/pilot/synthetic_null.yaml \
  outputs/pilot/synthetic_null
```

These commands only generate data and QC. They do not train a model, submit a
scaling sweep, or fit a scaling law.
