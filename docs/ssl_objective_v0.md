# Pilot v0 focal-cell SSL objective

The Pilot v0 molecular-only baseline predicts masked continuous expression for
one synthetic cell from that cell's remaining expression. It receives no
coordinates, neighbors, section identity, synthetic latent state, loadings, or
gene-class labels.

## Input and target

For every cell, a deterministic subset of genes is replaced by `0.0` after the
mask is selected. A binary visibility vector is concatenated with this masked
expression before the MLP encoder. The visibility vector distinguishes a
masked entry from a genuine observed zero. The target remains the unmodified
raw expression vector, and mean squared error is computed only at masked
positions.

Gene classes from `genes.csv` are used only to aggregate evaluation MSE for
intrinsic, mixed, and spatial targets. They never enter masking, model inputs,
or training-time feature construction.

## Determinism and splits

The corpus loader addresses only sections assigned to its requested split in
`metadata.json`; it never constructs a cell-level split. It caches a small
number of decompressed section shards rather than materializing the corpus.

Each cell's mask is generated from a stable BLAKE2b seed over the experiment
mask seed, mask epoch, and cell ID. It therefore does not depend on batch order,
worker order, or other cells. Evaluation always uses mask epoch zero and fixed
observation and masking seeds. Training advances the mask epoch on a configured
step interval.

## Masking regimes

`random` masks an exact rounded fraction of genes using an unbiased per-cell
permutation. Calibration fractions are 15%, 30%, 50%, and 70%.

`high_random` uses the identical unbiased mechanism but requires at least 50%
masking. Pilot calibration uses 85%. Its additional difficulty therefore comes
only from withholding more focal molecular information. It does not exploit
gene ordering, correlation estimates, synthetic classes, or latent ground
truth.

The calibration varies mask fraction while holding model architecture,
optimizer, train and validation sections, number of optimization steps,
training batch size, and evaluation observations fixed. This is an objective
calibration, not a scaling-law fit or hyperparameter search.

