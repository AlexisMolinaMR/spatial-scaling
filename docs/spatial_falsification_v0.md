# Pilot v0 spatial-context falsification

## Scientific question

This experiment asks whether correctly organized local molecular context lowers
focal-cell masked-expression MSE relative to an architecture-matched control:

```text
delta_space = L_shuffled - L_spatial.
```

The observation unit and model token are synthetic cells. The independent
experimental and split unit is a synthetic section. Expression is continuous
synthetic molecular measurement; no real spatial technology is implied. The
target is raw focal-cell expression at deterministically masked genes. K is
fixed at 256, not swept. Train, validation, and test sections are the corpus's
existing section-level splits. Metrics are masked MSE overall and for the
evaluation-only intrinsic, mixed, and spatial gene classes.

## True neighborhood

For a focal cell, the true condition selects the 256 other cells in the same
section having the smallest squared Euclidean distance in the saved local
micrometre coordinate system. The focal cell is assigned infinite distance and
cannot select itself. Global cell offset breaks exact-distance ties. Boundary
cells use their closest available cells; no wrapping or cross-section lookup is
performed. The model receives each neighbor's offset from the focal cell in
micrometres, divided inside the model by the configured 200 micrometre scale.

## Shuffled control

For each section and shuffle seed, a seeded permutation and nonzero cyclic
shift create a bijective derangement from every focal-cell offset to a different
anchor offset in that section. For each focal target:

1. compute its true KNN identities and relative-coordinate slots;
2. find cells local to its assigned incorrect anchor;
3. remove the focal identity and every identity in the target's true KNN;
4. fill exactly K molecular context positions in increasing distance from the
   incorrect anchor;
5. place these expression vectors into the unchanged true-neighborhood
   relative-coordinate slots.

Thus true and shuffled inputs have identical K and identical relative geometry,
and all context cells remain in the focal section. The shuffled expression
context retains local molecular covariance around another location but has zero
identity overlap with the target's true KNN. The construction uses no latent
state, gene class, split label, or synthetic ground truth. A fixed seed gives a
fixed mapping; changing the seed changes the correspondence.

## Model and masking

One cell is one token. A shared encoder maps concatenated expression and gene
visibility to a cell embedding. Context visibility is all one because context
expression remains observed in this first focal-target experiment. A learned
MLP embeds relative 2D offsets. Two dense transformer encoder blocks contextualize
the focal plus 256 context tokens, and a prediction head reads only the focal
token. Dense local attention is used for this K=256 correctness test; the
context encoder is modular so a later sparse implementation need not alter the
data interface.

The default contextual architecture is:

```text
embedding dimension: 64
layers: 2
attention heads: 4
FFN width: 128
dropout: 0
coordinate scale: 200 um
```

Focal masking retains the validated 30% `random` and 85% `high_random`
definitions. Contextual arms within a seed use identical initialization,
focal-mask realizations, optimizer, steps, batches, sections, and evaluation
observations. Gene classes are used only after prediction for metric
aggregation. The focal-only MLP is a reference with different capacity and is
not the decisive spatial control.

## Experiment matrix

Seeds are 101, 202, and 303. The positive corpus runs focal, shuffled, and
spatial conditions at both 30% and 85% masking. The null corpus runs shuffled
and spatial conditions at 85%. Every run uses AdamW, learning rate 3e-4,
weight decay 1e-4, and 2,000 optimization steps. Contextual runs use batch size
32; focal reference runs retain their validated batch size. Evaluation uses a
fixed 8,192-cell validation subset. Evaluation observation and exact-mask
SHA-256 hashes are saved with every run.

The GPU array is defined in `slurm/train_spatial_falsification.sbatch`. It
contains exactly 24 tasks and does not perform a K sweep. After all tasks finish,
aggregate them with:

```bash
uv run python scripts/summarize_spatial_falsification.py
```

The summary validates matched configurations, parameter counts, observations,
and masks before computing per-seed paired deltas and their mean and sample
standard deviation.
