# Project Mission

## Spatial Scaling

### Empirical scaling laws for spatial biological models

## 1. Scientific problem

Machine learning for spatial transcriptomics is moving toward increasingly large datasets and increasingly expressive models.

However, the field currently lacks a clear empirical understanding of what happens when these systems scale.

In language models, scaling can often be described primarily through quantities such as:

* number of training tokens;
* model parameters;
* training compute.

Spatial biology is fundamentally different.

A training example is embedded inside a hierarchy:

```text
molecule
↓
cell
↓
local neighborhood
↓
tissue region
↓
tissue section
↓
organ
↓
donor
↓
biological population
```

Consequently, one million additional cells can represent very different amounts of new information.

For example:

```text
1,000,000 cells
from one tissue section
```

and

```text
1,000,000 cells
from 100 tissues
and 100 donors
```

have identical raw cell counts but radically different biological coverage.

The central mission of Spatial Scaling is therefore:

> Determine the empirical principles governing how spatial biological models improve as data, biological diversity, spatial context, model capacity, data quality, and compute increase.

---

# 2. Central hypothesis

Raw observation count is an incomplete measure of scale for spatial biology.

We hypothesize that model performance depends on several partially independent dimensions:

[
S =
(
N_{\mathrm{cells}},
N_{\mathrm{samples}},
N_{\mathrm{donors}},
N_{\mathrm{tissues}},
N_{\mathrm{celltypes}},
G,
R,
Q
)
]

where:

* (N_{\mathrm{cells}}) = number of cells or spatial units;
* (N_{\mathrm{samples}}) = independent tissue sections or experiments;
* (N_{\mathrm{donors}}) = biological individuals;
* (N_{\mathrm{tissues}}) = tissue diversity;
* (N_{\mathrm{celltypes}}) = cellular diversity;
* (G) = molecular feature coverage;
* (R) = available spatial context;
* (Q) = measurement quality.

Model capacity (P) and compute (C) interact with these data dimensions.

We therefore expect spatial scaling to be better described as:

[
L = f(S,P,C)
]

rather than simply:

[
L = f(N_{\mathrm{cells}})
]

A major goal is to determine which components of (S) explain generalization.

---

# 3. What counts as a scaling law?

A scaling experiment must contain multiple controlled scale points.

A typical relationship might be:

[
L(N)=L_{\infty}+AN^{-\alpha}
]

where:

* (L(N)) is validation loss at scale (N);
* (L_{\infty}) represents an irreducible floor;
* (A) controls the magnitude;
* (\alpha) is the scaling exponent.

The exact functional form should not be assumed in advance.

Candidate models should be compared empirically.

Potential behaviors include:

* power-law scaling;
* log-linear scaling;
* saturation;
* broken power laws;
* threshold effects;
* interactions between dimensions.

A central question is whether different spatial tasks share scaling exponents or exhibit fundamentally different regimes.

---

# 4. Scaling dimensions

## 4.1 Number of cells

The simplest axis is:

[
N_{\mathrm{cells}}
]

Controlled subsampling can determine how prediction or representation quality changes as more cells become available.

However, cell-level scaling should initially keep biological composition approximately fixed.

This isolates the effect of repeated observations from the effect of added biological diversity.

---

# 4.2 Number of independent samples

Cells within the same tissue section are correlated.

We therefore distinguish:

[
N_{\mathrm{cells}}
]

from:

[
N_{\mathrm{samples}}
]

where a sample may represent a:

* tissue section;
* slide;
* biopsy;
* field group;
* experimental replicate.

This allows us to ask whether 100× more cells from the same specimen provide equivalent information to additional independent specimens.

We expect they will not.

---

# 4.3 Donor diversity

Spatial transcriptomics datasets frequently contain many cells but relatively few donors.

We will directly test:

[
N_{\mathrm{donors}}
]

while controlling approximate cell count.

This asks whether biological generalization is limited by cellular observations or by the number of independent biological individuals represented during training.

---

# 4.4 Tissue diversity

We will vary:

[
N_{\mathrm{tissues}}
]

while controlling total observation count where possible.

This provides a direct test of whether heterogeneous tissue coverage improves transfer to new biological environments.

Potential experiments include:

```text
same cell count
+
1 tissue
2 tissues
4 tissues
8 tissues
...
```

Evaluation should include held-out tissue settings.

---

# 4.5 Cellular diversity

Another relevant quantity is the diversity of cellular states represented during training.

Possible measurements include:

* annotated cell-type count;
* effective cell-state diversity;
* transcriptomic cluster diversity;
* entropy over cellular populations.

The number of cells and diversity of cells should be treated separately.

---

# 4.6 Molecular resolution

Models can receive different numbers of measured genes or molecular features.

We can define:

[
G = \text{number of molecular features}
]

and study scaling with:

* top HVGs;
* targeted panels;
* whole transcriptome;
* RNA plus protein;
* additional molecular modalities.

Possible scale points might include:

```text
128 genes
256
512
1,024
2,048
4,096
...
```

This allows us to determine whether spatial context becomes more or less valuable as molecular resolution changes.

---

# 4.7 Spatial context

Spatial data provides a distinct scale dimension.

For a cell (i), the model may observe a neighborhood:

[
\mathcal{N}_r(i)
]

defined by physical radius (r), or:

[
\mathcal{N}_k(i)
]

defined by (k) nearest neighbors.

We can scale:

* physical radius;
* number of neighbors;
* graph hops;
* number of spatial tokens;
* tissue-region context.

This produces a spatial receptive-field scaling curve.

A key question is whether there exists a characteristic biological scale beyond which additional context provides little useful information.

---

# 4.8 Model capacity

Models should be evaluated over parameter scales such as:

[
P_1 < P_2 < \dots < P_n
]

Model scaling should ideally preserve the architectural family so parameter count can be interpreted cleanly.

Potential variables include:

* hidden dimension;
* layer count;
* attention heads;
* graph layers;
* MLP width.

---

# 4.9 Compute

Data and model scale cannot be interpreted independently of training compute.

We should record at minimum:

* optimizer steps;
* examples or tokens processed;
* parameter count;
* GPU hours;
* hardware type.

Where practical, estimate training FLOPs.

This enables experiments asking whether additional compute should be spent on:

* larger models;
* more cells;
* more biological diversity;
* larger spatial context.

---

# 5. Spatial representation is an experimental variable

There is currently no reason to assume one representation of a tissue is universally correct.

The project should therefore compare several spatial organizations.

## Cell tokens

Each cell is represented as a token.

This gives a natural biological unit but requires defining interactions between cells.

## Spot tokens

For technologies without single-cell resolution, measurement spots may act as tokens.

These should not be conflated with actual cells.

## Neighborhood tokens

A token can represent a local collection of cells.

This may allow larger physical context using fewer sequence elements.

## Graph representations

Cells may form nodes in graphs defined using:

* k-nearest neighbors;
* radius thresholds;
* Delaunay triangulation;
* tissue boundaries.

## Multi-hop context

For graph-based systems we can explicitly test:

```text
1 hop
2 hops
3 hops
...
```

This creates a clean receptive-field scaling axis.

## Hierarchical tissue representations

Spatial organization may be hierarchical:

```text
cell
→ neighborhood
→ tissue domain
→ whole section
```

Hierarchical models may scale differently from flat cell-token models.

---

# 6. Cell-cell communication

Geometric proximity and biological communication are not equivalent.

The project should therefore distinguish:

### Geometric graph

Edges determined only from coordinates.

### Molecular interaction graph

Edges weighted using biological information such as ligand-receptor compatibility.

### Learned interaction graph

Edges inferred directly by the model.

We can compare these representations under matched data and model budgets.

This may reveal whether explicit biological priors primarily improve low-data regimes or continue to provide advantages at large scale.

---

# 7. Data quality as a scaling dimension

Real spatial datasets vary substantially in quality.

Rather than treating this only as preprocessing noise, we will study it explicitly.

Potential perturbations include:

## Gene measurement quality

Simulate:

* lower count depth;
* increased dropout;
* reduced gene panels;
* increased measurement noise.

## Spatial coordinate quality

Perturb coordinates by:

[
x'_i = x_i + \epsilon_i
]

where (\epsilon_i) is controlled spatial noise.

## Segmentation quality

Introduce controlled:

* boundary errors;
* merged cells;
* split cells;
* missing cells.

## Alignment quality

For datasets combining histology, molecular measurements, or serial sections, perturb registration systematically.

This gives:

[
Q_{\mathrm{alignment}}
]

as an explicit experimental variable.

We can then ask whether poor alignment changes the scaling exponent or merely shifts performance downward.

---

# 8. Biological generalization

Scaling laws are only useful if the evaluation regime is clearly specified.

We should distinguish several settings.

## Within-sample

Held-out cells from tissue samples observed during training.

This is the easiest setting and should not be interpreted as strong biological generalization.

## Held-out donor

Train on some individuals and evaluate on unseen individuals.

## Held-out tissue

Evaluate on a tissue absent from training.

## Held-out condition

Evaluate on unseen:

* diseases;
* treatments;
* developmental states.

## Held-out technology

Train on one or more spatial technologies and evaluate transfer to another.

Scaling behavior may differ substantially across these regimes.

---

# 9. Baseline hierarchy

The project should establish a progression from simple to complex models.

## Baseline 0 — molecular only

Model individual cells without coordinates.

This establishes how much performance is obtainable without spatial information.

## Baseline 1 — simple neighborhood pooling

Aggregate molecular information from nearby cells.

Examples:

* mean pooling;
* distance-weighted pooling.

## Baseline 2 — graph neural network

Represent the tissue as a spatial graph.

## Baseline 3 — attention-based spatial model

Use cells or neighborhoods as tokens.

## Baseline 4 — hierarchical spatial architecture

Represent multiple spatial scales explicitly.

Only after establishing this hierarchy should we consider substantially more complex foundation-model architectures.

---

# 10. Experimental matrix

The ideal paper should contain multiple orthogonal scaling studies.

A possible matrix is:

| Experiment        | Varied                        | Controlled         |
| ----------------- | ----------------------------- | ------------------ |
| Cell scaling      | cells                         | tissue composition |
| Sample scaling    | sections                      | cells              |
| Donor scaling     | donors                        | cells              |
| Tissue scaling    | tissue diversity              | cells              |
| Gene scaling      | molecular features            | cells              |
| Context scaling   | spatial radius/hops           | cells/model        |
| Model scaling     | parameters                    | data               |
| Compute scaling   | training budget               | architecture       |
| Quality scaling   | noise/measurement quality     | data quantity      |
| Alignment scaling | coordinate/registration error | model/data         |
| Tokenization      | representation                | data/model budget  |

This structure is important because many apparent scaling effects are otherwise confounded.

---

# 11. Effective spatial data scale

One ambitious outcome would be an empirical measure of effective spatial dataset size.

Instead of:

[
N_{\mathrm{effective}} = N_{\mathrm{cells}}
]

we might find a relationship such as:

[
N_{\mathrm{effective}}
======================

N_{\mathrm{cells}}^{a}
N_{\mathrm{donors}}^{b}
N_{\mathrm{tissues}}^{c}
Q^{d}
D_{\mathrm{spatial}}^{e}
]

or another empirically supported formulation.

The purpose is not to force this particular equation.

The scientific goal is to determine whether a low-dimensional description of dataset scale can predict performance across different spatial corpora.

Such a result could provide practical guidance for:

* dataset construction;
* model design;
* compute allocation;
* experimental data generation.

---

# 12. Expected outputs

The project should ultimately produce:

## Scaling curves

Performance versus:

* cells;
* samples;
* donors;
* tissues;
* genes;
* spatial context;
* model parameters;
* compute.

## Scaling exponents

Where power-law-like behavior is supported.

## Saturation regimes

Identify dimensions where additional scale provides diminishing returns.

## Interaction maps

Determine interactions such as:

[
\text{model size} \times \text{data diversity}
]

or:

[
\text{spatial context} \times \text{data quality}
]

## Practical recommendations

For a fixed experimental or compute budget:

> Is it more useful to collect more cells, more donors, more tissues, higher-quality measurements, or train a larger model?

This is one of the most practically important outputs of the project.

---

# 13. What would make the paper scientifically strong?

A strong paper should not rely on one architecture or one dataset.

The strongest version would demonstrate that several reproducible empirical phenomena hold across:

* multiple datasets;
* multiple tissues;
* multiple spatial technologies;
* more than one model family.

The contribution should therefore be the discovery of properties of **spatial biological learning**, rather than the introduction of another spatial architecture.

---

# 14. What the project is not

The project is not primarily intended to:

* introduce the largest spatial foundation model;
* optimize one benchmark;
* claim biological understanding from attention weights;
* compare architectures without controlling data scale;
* treat every cell as an independent training example;
* assume physical proximity equals biological communication.

These may become components of individual experiments, but they are not the central scientific contribution.

---

# 15. Mission

The final objective of Spatial Scaling is to establish an empirical science of scale for spatial biological machine learning.

We want to answer:

> How much data do spatial models need?

But also:

> What kind of data?

> From how many tissues and donors?

> At what molecular and spatial resolution?

> Using what neighborhood structure?

> At what measurement quality?

> With how much model capacity and compute?

And ultimately:

> Which dimensions of scale determine genuine generalization to new biological systems?
