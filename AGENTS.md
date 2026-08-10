# AGENTS.md

## Project objective

Spatial Scaling studies empirical scaling laws for machine-learning models
trained on spatial transcriptomics and related spatial molecular data.

The goal is not merely to show that larger models or larger datasets perform
better. The goal is to identify which dimensions of scale determine learning,
transfer, and biological generalization.

Relevant scaling dimensions include:

* cells, spots, and spatial tokens;
* independent samples and tissue sections;
* donors and biological contexts;
* tissue, cell-type, and cell-state diversity;
* molecular feature coverage;
* spatial receptive field and neighborhood structure;
* model capacity;
* training compute;
* measurement, segmentation, coordinate, and alignment quality.

Read `docs/project_mission.md` before changes affecting the scientific
formulation, data model, scaling variables, spatial representation, training
objective, evaluation protocol, or biological interpretation.

Do not invent scientific details not defined in the repository or task.

Before a scientifically meaningful change, identify:

* input modality and spatial technology;
* biological and experimental unit;
* model input unit or token;
* target or learning objective;
* scaling variable;
* quantities intended to remain controlled;
* grouping variables;
* train/validation/test split;
* metric;
* relevant confounders.

State material assumptions.

## Scientific principles

Do not call a result a scaling law merely because performance improves with
more data or parameters.

A scaling experiment should evaluate a relationship between a defined scale
variable and an outcome across multiple scale points.

Whenever practical, report:

* all scale points;
* uncertainty across seeds or resampling;
* cells, spots, or tokens used;
* independent sample count;
* donor and tissue diversity;
* parameter count;
* training budget and compute;
* data-quality statistics;
* in-distribution performance;
* held-out biological generalization.

Do not equate:

* cell count with independent sample count;
* cell count with biological diversity;
* spatial proximity with biological communication;
* random-cell validation with biological generalization;
* parameter count with compute;
* training tokens with independent biological information.

Prefer controlled experiments that vary one interpretable factor at a time.
When this is impossible, record the confounding explicitly.

## Environment and validation

* Use `uv sync` to create or update the environment.
* Run project commands with `uv run`.
* Run `./scripts/check.sh` before completing code changes.
* If it does not yet exist, run at minimum:

```bash
uv run pytest
uv run ruff check .
```

* Do not report checks as passing unless they were executed.
* Do not change dependency manager or lockfile unless required.
* Do not add dependencies without a concrete need.
* Do not introduce Conda, Poetry, or another environment manager.

## Repository conventions

* Reusable code: `src/spatial_scaling/`
* Tests: `tests/`
* Scientific documentation: `docs/`
* Experiment configs: `configs/`
* Cluster scripts: `slurm/`
* Thin executable wrappers: `scripts/`

Configuration should be explicit rather than hard-coded.

Add or update tests for behavior changes.

Make small, focused changes.

Do not refactor or reformat unrelated files.

Preserve public APIs, schemas, manifests, identifiers, and output formats unless
the requested change requires otherwise.

Core results should be reproducible from code and configuration rather than
depending on notebook state.

## Scientific safety

Do not silently alter:

* gene, cell, spot, donor, patient, or sample identifiers;
* tissue, disease, cell-type, technology, or batch labels;
* slide, section, or field identifiers;
* spatial coordinates or coordinate units;
* expression units or feature definitions;
* controls;
* splits;
* scaling variables;
* metric definitions.

Preserve raw source identifiers when normalized identifiers are introduced.

Fail clearly when required identifiers, metadata, units, coordinates, features,
or grouping variables are missing.

Keep source annotations, normalized annotations, derived annotations, and model
predictions distinct.

Randomized operations must accept an explicit seed.

Keep filtering, exclusions, preprocessing, sampling, scale-point generation,
and metric definitions traceable.

Report exclusions rather than silently dropping observations.

Do not interpret model outputs, associations, embeddings, learned edges, or
attention patterns as biological conclusions unless supported by the study
design and computed results.

## Scaling experiments

Every scaling experiment must define:

1. scale variable;
2. scale points;
3. sampling procedure;
4. controlled variables;
5. evaluation distribution;
6. metric;
7. training budget;
8. randomization procedure.

Possible scale axes include:

### Data quantity

* cells;
* spots;
* spatial tokens;
* tissue sections;
* independent samples.

### Biological diversity

* donors;
* tissues;
* organs;
* diseases;
* cell types;
* cell states.

### Molecular scale

* genes;
* HVGs;
* panel size;
* number of modalities.

### Spatial scale

* neighbors;
* physical radius;
* graph hops;
* receptive field;
* hierarchical tissue depth.

### Model scale

* parameters;
* width;
* depth;
* attention depth;
* graph depth.

### Compute scale

* optimization steps;
* tokens processed;
* GPU hours;
* estimated FLOPs where practical.

### Data quality

* sequencing depth;
* missingness;
* feature dropout;
* spatial resolution;
* segmentation noise;
* coordinate noise;
* registration error.

Do not describe experiments with changing biological composition as pure
cell-count scaling.

Do not compare model sizes trained with substantially different budgets without
recording the difference.

Do not choose scale points after inspecting final test performance.

## Biological dependence and leakage

Spatial observations are correlated.

Cells or spots from the same:

* local neighborhood;
* field of view;
* tissue region;
* section;
* slide;
* donor;
* patient;
* batch;
* organ;
* condition

must not automatically be treated as independent.

Define the generalization question before constructing splits.

Possible regimes include:

* held-out cells;
* held-out spatial regions;
* held-out sections or slides;
* held-out donors;
* held-out tissues;
* held-out diseases or conditions;
* held-out technologies.

Random cell splits are not evidence of donor-, tissue-, or technology-level
generalization.

Prevent replicates, near duplicates, adjacent duplicated crops, and shared
held-out experimental units from leaking across partitions.

Fit learned preprocessing on training data only unless a transductive protocol
is explicit.

Record grouping variables used for every split.

## Spatial representation

Do not prematurely treat one representation as canonical.

Potential model units include:

* cells;
* spots;
* image or molecular patches;
* local neighborhoods;
* tissue domains;
* hierarchical regions.

Potential graphs include:

* k-nearest-neighbor;
* radius;
* Delaunay;
* experimentally defined adjacency.

Do not treat spots as equivalent to cells.

Do not assume spatial distance alone represents biological interaction.

Keep neighborhood construction explicit and configurable.

Coordinate systems and physical units must remain traceable.

Before combining datasets, determine whether coordinates are expressed in
pixels, micrometers, local coordinates, global slide coordinates, or registered
coordinates.

Do not combine incompatible coordinate systems without explicit conversion.

## Cell-cell communication

Keep geometric proximity separate from inferred biological communication.

Distinguish:

* geometric adjacency;
* molecularly informed adjacency;
* learned adjacency.

Do not silently use ligand-receptor priors in a geometry-only baseline.

Do not call a learned edge biological communication without independent
evidence.

## Modeling safety

Do not assume a transformer, graph neural network, or any other architecture is
required.

Do not assume cells are the optimal tokenization.

Keep these components modular:

* dataset representation;
* tokenization;
* neighborhood construction;
* model;
* training;
* scaling sweep;
* evaluation.

Maintain simple baselines before increasing model complexity.

When studying spatial information, maintain a molecular-only baseline.

Avoid changing architecture during a scaling sweep unless architecture is
itself the experimental variable.

Prefer the baseline hierarchy:

1. molecular-only model;
2. simple neighborhood aggregation;
3. graph model;
4. attention-based spatial model;
5. hierarchical or multi-scale model.

## Data and provenance

* Do not commit raw data, large matrices, spatial images, checkpoints, secrets,
  logs, caches, runs, or large generated outputs.
* Store heavy datasets outside the Git repository.
* Treat downloaded source data as immutable.
* Do not delete or overwrite raw source data.
* Prefer configurable data roots over scattered hard-coded cluster paths.
* Validate schemas, dtypes, shapes, missing values, coordinates, identifiers,
  and uniqueness constraints at data boundaries.
* Preserve provenance from derived examples back to source datasets and
  biological samples.
* Keep manifests, checksums, inventories, and small metadata in the repository
  when useful.
* Do not download datasets or checkpoints, or clone external repositories,
  unless explicitly requested.
* Tests must use synthetic data or small fixtures and must not require network
  access.

When harmonizing datasets, preserve technology-specific metadata.

Do not assume different technologies measure equivalent spatial units.

Any gene intersection, vocabulary mapping, normalization, or coordinate
rescaling must be explicit and reproducible.

## Compute and cluster safety

* Do not run expensive preprocessing, training, or GPU workloads on login nodes.
* Do not submit scheduler jobs unless explicitly requested.
* Do not assume partition, account, GPU, memory, node, or time-limit settings.
* Use dry runs or reduced-data smoke tests before expensive workflows when
  practical.
* Scaling sweeps should be restartable.
* Do not silently overwrite completed results.
* Separate sweep generation, submission, training, aggregation, and scaling-law
  fitting.
* Job generation must not implicitly submit jobs.
* Record the Git commit for important experiments when practical.

## Evaluation

State what is being generalized before constructing a split.

Distinguish cell-, region-, section-, sample-, donor-, tissue-, disease-, and
technology-level generalization.

Prefer held-out biological units when the scientific claim concerns transfer.

Compare against simple baselines.

Keep metric computation separate from plotting and interpretation.

Do not select metrics solely because they favor one model.

For scaling-law fitting:

* report fit uncertainty;
* report goodness of fit;
* show observed scale points;
* compare plausible functional forms;
* do not assume a power law by default;
* avoid strong extrapolation beyond the observed scale range.

Candidate behaviors include power laws, log-linear trends, saturation, broken
power laws, and threshold effects.

## Plotting

Plots are scientific outputs and must be treated as part of the analysis, not
as decoration.

Every figure should make the underlying experiment auditable.

General rules:

* Use matplotlib or seaborn with simple readable defaults.
* Keep plot text black.
* Use clear axis labels including units where applicable.
* Use short descriptive titles.
* Avoid decorative styling, unnecessary gradients, 3D effects, or visual clutter.
* Use consistent naming, ordering, and scales across related figures.
* State what error bars or shaded regions represent.
* Show sample counts when they materially affect interpretation.
* Do not encode conclusions into titles.
* Do not add significance stars or statistical annotations unless requested.
* Do not hide failed seeds, anomalous scale points, or outliers without a
  documented exclusion rule.
* Keep metric computation separate from plotting code.
* Save figure-generation inputs or make them reproducible from saved results.

For scaling figures:

* show the observed scale points;
* show individual seeds when practical;
* distinguish observations from fitted curves;
* distinguish interpolation from extrapolation;
* use log axes only when scientifically justified;
* label logarithmic axes explicitly;
* report the fitted relationship and uncertainty separately from the visual;
* do not infer a power law from a visually straight log-log plot alone;
* keep identical axis limits when comparing panels where visual comparison is
  part of the claim.

For comparisons across tissues, donors, technologies, or models:

* preserve category order across figures;
* avoid arbitrary color changes between related plots;
* avoid overloaded legends;
* use facets or separate plots when a single figure becomes difficult to read.

Figures used in reports or papers should be reproducible from analysis outputs
without manually editing plotted values.

## Git safety

* Do not commit or push unless explicitly requested.
* Do not force-push.
* Do not use destructive Git commands.
* Do not discard changes made by another person or agent.
* Inspect the branch and working tree before editing.
* Keep changes compatible with parallel agent work whenever practical.

## Development priorities

Prefer this order unless the task requires otherwise:

1. define datasets and provenance;
2. define biological and experimental units;
3. define canonical internal data structures;
4. establish leakage-safe evaluation;
5. implement molecular-only baselines;
6. implement simple spatial baselines;
7. define scale-point generation;
8. study cell and sample scaling;
9. study donor and tissue-diversity scaling;
10. study spatial tokenization and receptive field;
11. study molecular-resolution scaling;
12. study quality and alignment scaling;
13. study model and compute scaling;
14. fit empirical scaling relationships;
15. test reproducibility across datasets and technologies.

Do not allow architecture development to outrun data handling, evaluation, and
baseline quality.

## Completion report

For completed implementation tasks, report:

* files changed;
* commands executed;
* validation results;
* assumptions made;
* scientific decisions introduced;
* implementation decisions introduced;
* datasets, splits, or scaling definitions affected;
* unresolved limitations;
* whether cluster jobs were submitted.
