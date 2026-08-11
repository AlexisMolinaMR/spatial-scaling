# Pilot v0 synthetic spatial-scale calibration

## Scientific distinction

The completed K = 32...1024 experiment found a nearly flat positive-world
spatial advantage even though its median Kth-neighbor radius crossed the
generator's 200 µm RBF scale. This diagnostic separates two quantities that
must not be conflated:

```text
K_context = number of observed context cells
R_context = physical geometry of those observed cells.
```

A fixed K can span different radii as density changes. Conversely, increasing
K can add redundant observations within an already informative local field.
Neither K nor the point where a K curve flattens is identified as a spatial
correlation length.

The model token remains a synthetic cell, the objective remains focal-cell
masked expression prediction, and the split unit remains the synthetic tissue
section. The positive and null corpora, 85% high-random masking, gene classes,
architecture, optimizer, 2,000-step budget, batch size 32, and seeds 101, 202,
and 303 are unchanged. No scaling law or correlation length is fitted.

## Experiment A: low K

The executable grid adds K = 1, 2, 4, 8, and 16 under matched spatial and
shuffled conditions. The exactly compatible completed K = 32 and K = 64 runs
from `pilot_context_scaling_v0` are reused rather than retrained. Their source
experiment IDs, configuration identities, Git SHA, and run paths are written
to `reuse_provenance.json` by the aggregator.

## Experiment B: distance exclusion

At fixed K = 32, a focal target first excludes every candidate with physical
Euclidean distance below `d_min`. The selector then returns the closest 32
remaining cells in the same section. The focal cell is always excluded,
including at `d_min = 0`. Distance and cell offset define deterministic order.
An observation with fewer than 32 valid cells fails selection; K is never
reduced and contexts are never padded.

The positive thresholds are 0, 50, 100, 150, 200, 300, 400, and 600 µm. The
null thresholds are 0, 200, and 600 µm. These thresholds change molecular
source identities, not merely coordinate embeddings.

## Matched shuffled-distance control

The existing seeded bijective same-section target-to-wrong-anchor derangement
is retained. For each target:

1. true relative-coordinate slots come from its distance-excluded 32-cell
   context;
2. molecular candidates around the wrong anchor must independently satisfy
   `distance(anchor, candidate) >= d_min`;
3. the wrong anchor, focal identity, and every true selected-context identity
   are excluded;
4. the nearest 32 remaining wrong-anchor candidates populate the unchanged
   true coordinate slots.

Thus spatial and shuffled arms retain identical target observations, focal
masks, K, coordinate-slot dimensions, model, initialization, optimizer, and
training budget while breaking focal-location-to-molecular-context
correspondence.

## Eligibility and comparability

For every distance condition, focal observations are drawn from the exact
matched eligibility intersection defined at the maximum threshold, 600 µm,
for that split and seed. A target belongs to this intersection only if both
the true and shuffled selectors can return exactly K = 32 cells. The same
subset is therefore used at all lower thresholds and by both conditions.

The fixed deterministic validation observations are filtered without
backfilling. Training samples only from the common eligible offsets in each
section. Counts before filtering, at the requested threshold, and in the
common subset are recorded. If all cells are eligible, the original sampling
path is preserved exactly.

Eligibility uses a deterministic set of extreme-coordinate witnesses only to
prove a lower bound on the number of valid candidates. Any target that cannot
be certified by that lower bound is checked with the full exact selectors.
The resulting policy is exact, not approximate.

## Realized physical distances

For each evaluation target, the harness measures the nearest, median, and
farthest selected true-context distances. Each distribution is summarized by
its median, P25, P75, P05, and P95. These measurements use saved physical
micrometre coordinates and are identical across a matched spatial/shuffled
pair. The configured 200 µm RBF scale is shown only as a descriptive reference.

## Specification and execution

`configs/pilot/spatial_scale_calibration_v0.yaml` is the single source of
truth. The existing manifest generator records K, `d_min`, common eligibility
threshold, corpus, condition, seed, full training configuration, collision-safe
experiment ID, output path, and joint configuration-selection SHA-256.

GPU tasks use `slurm/train_context_scaling.sbatch`. Final aggregation uses
`slurm/summarize_spatial_scale_calibration.sbatch` and writes seed-level paired
deltas before mean and sample-standard-deviation summaries. The aggregation
also produces raw-loss, null-control, low-K, threshold-distance, and realized-
distance plots.

## Measured outcome

SLURM jobs 151288, 151293, 151327, and 151332 completed all 96 new runs
without failure. The 12 completed K = 32 and K = 64 runs were reused with
their original provenance. The `d_min = 0` gate reproduced every prior K = 32
raw validation loss and paired delta exactly.

Low-K spatial-gene advantage was already 0.4535 ± 0.0038 at K = 1, compared
with 0.4599 ± 0.0058 at K = 32. Mixed-gene advantage was 0.2178 ± 0.0007 at
K = 1 and 0.2267 ± 0.0037 at K = 32. Thus the original flat K curve does not
hide a substantial rise between K = 1 and K = 32.

Distance exclusion resolved the planted physical scale descriptively. Mean
spatial-gene advantage decreased from 0.4599 at realized median selected-cell
distance 50.6 µm to 0.3329 at 158.9 µm, 0.2267 at 207.0 µm, 0.0675 at
305.1 µm, and 0.0090 at 404.2 µm. At 603.4 µm it was slightly negative
(-0.00645). Mixed genes followed the same lower-magnitude pattern. Intrinsic
advantages stayed close to zero. Null spatial-gene advantages remained within
approximately 0.00002 of zero at all three tested thresholds.

Every train and evaluation cell was eligible even at 600 µm, so no focal
population changed with threshold: 393,216 train cells, 65,536 validation-split
cells, and all 8,192 requested validation observations were retained. The
distance dependence is therefore not an eligibility-population artifact.

These measurements support the local-redundancy hypothesis: one extremely
nearby cell is already highly informative, while removing nearby molecular
observations reveals a physical decay broadly comparable to, but not equated
with, the configured 200 µm RBF scale. No correlation length or scaling law is
fitted. The synthetic spatial calibration stage passes and is sufficiently
understood to proceed to a separately designed N_train scaling stage.
