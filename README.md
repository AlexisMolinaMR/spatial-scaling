# Spatial Scaling

Empirical scaling laws for spatial transcriptomics models.

Pilot v0's controlled synthetic spatial-context sweep and its low-K/
distance-exclusion calibration are documented in
`docs/context_scaling_v0.md` and `docs/spatial_scale_calibration_v0.md`.

Spatial representation learning is rapidly adopting larger datasets and increasingly complex architectures, but it is not yet clear what actually drives improvements in spatial biological models.

This project studies how model performance changes as we independently scale:

* training data;
* biological diversity;
* model capacity;
* spatial context;
* molecular resolution;
* measurement quality;
* alignment quality;
* compute.

The central goal is to determine whether spatial biological models exhibit predictable scaling behavior and, more importantly, **what kind of scale matters**.

## Motivation

Conventional machine-learning scaling laws typically relate model loss to quantities such as:

[
N = \text{number of training examples}
]

[
P = \text{number of model parameters}
]

[
C = \text{training compute}
]

Spatial transcriptomics introduces additional structure.

A dataset with ten million cells from one tissue is not necessarily equivalent to ten million cells distributed across many tissues and donors.

Likewise, increasing a model's receptive field can mean very different things biologically depending on whether spatial structure is represented using:

* neighboring cells;
* graph hops;
* physical distance;
* tissue regions;
* inferred cell-cell communication.

Spatial Scaling therefore treats dataset scale as a multidimensional quantity rather than a single number.

## Core questions

We want to answer questions such as:

**Data scaling**

How does performance change with the number of cells, spots, tissue sections, or spatial tokens?

**Biological diversity**

Is adding new tissues, donors, cell types, or disease contexts more valuable than adding additional cells from an existing context?

**Spatial information**

When does explicit spatial context improve over models that only observe molecular profiles?

**Spatial receptive field**

How much neighborhood context is useful?

Does performance improve with:

* more neighbors;
* larger physical radii;
* additional graph hops;
* hierarchical tissue context?

**Tokenization**

What is the appropriate unit for a spatial foundation model?

Possibilities include:

* cells;
* spots;
* spatial patches;
* local neighborhoods;
* tissue regions.

**Data quality**

How sensitive is scaling to:

* sequencing depth;
* gene detection;
* segmentation errors;
* coordinate noise;
* registration errors;
* spatial resolution?

**Model scaling**

Do larger models consistently benefit from more spatial data?

At what point does performance become data-limited rather than model-limited?

**Transfer**

How do these relationships change when evaluating on unseen:

* donors;
* tissues;
* organs;
* diseases;
* technologies?

## Working hypothesis

Spatial models will not follow a useful scaling law based only on the number of cells.

We expect performance to depend on several effective dimensions of the training corpus, including:

[
N_{\text{cells}}
]

[
N_{\text{tissues}}
]

[
N_{\text{donors}}
]

[
D_{\text{molecular}}
]

[
R_{\text{spatial}}
]

[
Q_{\text{measurement}}
]

and model capacity (P).

One goal of the project is to determine whether these factors can be summarized by an **effective spatial data scale** that predicts downstream generalization better than raw sample count.

## Initial experimental strategy

We will begin with controlled experiments before attempting very large models.

### 1. Establish datasets

Build a corpus spanning multiple spatial transcriptomics technologies, tissues, donors, and spatial resolutions.

### 2. Establish non-spatial baselines

Measure what can be predicted from molecular measurements alone.

This determines whether explicit spatial information contributes anything beyond cell identity and expression state.

### 3. Establish simple spatial baselines

Represent local tissue context using methods such as:

* k-nearest-neighbor graphs;
* radius graphs;
* local neighborhood aggregation.

### 4. Scale data quantity

Train on controlled fractions of the available corpus.

Example:

```text
1%
2%
5%
10%
20%
40%
70%
100%
```

### 5. Scale biological diversity

Hold approximate cell count constant while changing the number of:

* tissues;
* donors;
* cell types;
* biological contexts.

### 6. Scale spatial context

Vary:

* neighborhood size;
* graph depth;
* physical radius;
* hierarchical context.

### 7. Scale model capacity

Evaluate multiple model sizes under comparable training conditions.

### 8. Perturb data quality

Systematically degrade:

* gene measurements;
* segmentation;
* spatial coordinates;
* registration;
* spatial resolution.

Measure whether larger datasets or larger models compensate for lower-quality measurements.

## Scaling models

A basic empirical relationship may take the form:

[
L(N) = L_{\infty} + A N^{-\alpha}
]

where:

* (L) is validation loss or error;
* (N) is dataset scale;
* (L_{\infty}) is an irreducible error term;
* (\alpha) is the scaling exponent.

For spatial biology, we may require multivariate relationships such as:

[
L =
f(
N_{\text{cells}},
N_{\text{tissues}},
N_{\text{donors}},
P,
R,
Q,
C
)
]

where (R) represents spatial context and (Q) represents measurement quality.

The project will determine empirically which parameterizations are justified.

## Repository structure

```text
spatial-scaling/
├── AGENTS.md
├── README.md
├── configs/
├── data/
├── docs/
│   └── project_mission.md
├── outputs/
├── scripts/
├── slurm/
├── src/
│   └── spatial_scaling/
└── tests/
```

## Development

The project uses `uv`.

Install the environment:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Pilot v0 includes a reproducible synthetic spatial-expression calibration
corpus. Its mathematical definition, output schema, QC, and generation commands
are documented in [`docs/pilot_v0.md`](docs/pilot_v0.md).

The focal-cell masked molecular objective, leakage controls, and deterministic
masking regimes are documented in
[`docs/ssl_objective_v0.md`](docs/ssl_objective_v0.md).

The K=256 true-spatial versus shuffled-context falsification design is
documented in
[`docs/spatial_falsification_v0.md`](docs/spatial_falsification_v0.md).

The manifest-driven spatial receptive-field sweep and computational profiling
protocol are documented in
[`docs/context_scaling_v0.md`](docs/context_scaling_v0.md).

## Status

Early research and infrastructure stage.

Current priorities are:

1. selecting the initial spatial datasets;
2. defining canonical data units;
3. choosing evaluation tasks;
4. implementing baseline spatial representations;
5. designing controlled scaling experiments.

The repository intentionally does not yet commit to a single spatial architecture.

## Scientific philosophy

The objective is not to demonstrate that a sufficiently large transformer can obtain a better benchmark score.

The objective is to identify the **empirical laws governing learning from spatial biological data**:

> What should we scale, how should we represent it, and what forms of scale produce genuine biological generalization?
