# Registered J0 benchmark protocol

## Task and permitted information

The task is **same-target, held-context transfer on shared measured genes**.
The five contexts are K562_essential, RPE1, HepG2, Jurkat and Orion_HCT116.
The released axes contain 1,525 perturbation targets and 5,979 output genes.
The 2,048 normalization anchors are a separate fixed coordinate set.

At prediction time, a method can use:

- The query perturbation target identifier.
- The measured encoder-control pool in the target context.
- Same-target perturbed cells and controls in the fold's three training contexts,
  when supported by that method's declared native interface.
- Generic pretrained gene representations and external STRING associations.

The target context's perturbed cells are evaluation labels, not model inputs.
The availability of measured target controls must be disclosed in every comparison.
Input-interface differences must also be disclosed: a comparator that does not
retrieve source responses at inference should not be described as doing so.

The benchmark is retrospective. Existing target/gene axes and prior exploration
do not establish a new, independently blinded biological test. This protocol does
not claim unseen-target or unseen-gene generalization.

## Fold generation

`cosit.splits.make_fold` hashes **identifiers only**:

```text
sha256("joint-condition-holdout-v1|17|FOLD_ID|TARGET_ID")
```

The first 459 targets in this order are split into three disjoint groups of 153,
assigned to the three training contexts. A held condition cannot enter training
as query supervision, source response, negative-bank truth or scale-fitting data.
Its same-target observations in the other training contexts may remain legal.
There are 3 × (1,525 − 153) = **4,116 query conditions per epoch**.

`configs/splits/` records the frozen folds, allowed training conditions and exact
validation panels. `cosit split --verify-frozen configs/splits` checks regeneration
against those records. Panel selection keeps the original coordinate order;
internal holdout axes are hash-ordered rather than lexically sorted.

Seeds 17, 29 and 43 define independent task initialization and training schedules.
They share the same partitions. Every model is trained from scratch; a checkpoint
is loaded only to resume that **same run** or to evaluate its selected model.

## Freshly fitted statistical assets

- Control PCA, feature-space compression and local connection frames use only
  the three training contexts' control statistics.
- Target-context controls provide lawful prediction background but do not fit
  those global geometry assets.
- Response scales and strength strata use only allowed training conditions.
- Source means, quantiles and bootstrap precisions are empirical observations,
  not reused predictions or task-trained feature caches.
- Source responses held from gradient training become available only to a
  separate `no_grad` inference-source capability during sealed evaluation.
- The same STRING graph and generic priors are independent of response labels.

The portable data store rejects forbidden requests **before** reading numeric
arrays and rechecks the permission on cache hits. File hashes and immutable file
identities protect observations; evaluator capabilities require completed,
sealed predictions. These checks prevent accidental leakage inside this code,
not malicious access by another Python program.

## Joint optimization

The original `BatchedSharedPathModel` and connection solver are retained under
`cosit.core`. Its main defaults are prior dimension 261, MLP width 128, joint state
rank 32, local action rank 4, feature ambient dimension 32, connection strength
0.15 and ridge 0.2. Every task module is trainable throughout phase J.

Training draws one query in each of three contexts in balanced groups. The
effective update contains up to 16 conditions per query context (48 total).
Each query retrieves one or two other lawful training sources, excluding any
source-held condition. Homogeneous query/source layouts are vectorized and
their losses accumulated with weights equal to their condition counts. Cells
belonging to different perturbations are never merged into one distribution.

Each condition uses a 64-cell generated cloud and a deterministic 64-cell draw
from its real training condition. Supervision moments use the complete immutable
condition cache (up to 256 cells), not merely this epoch's cell draw.

The five joint loss groups and weights are:

| Group | Weight | Role |
|---|---:|---|
| `joint_delta` | 1.00 | Scaled response error, correlation and strong-gene fit |
| `joint_discrimination` | 0.25 | Training-only, reliability-balanced cosine RankNet surrogate |
| `joint_marginal` | 1.00 | Differentiable hurdle W1, positive shape/detection/mean |
| `joint_structure` | 0.25 | Centered sliced W1, shrunk covariance and rank regularity |
| `joint_shared` | 0.05 | Target statistics, source reconstruction and source readout |

The exact formulas live in `core/objective.py` and are not replaced by a single
MSE objective. `joint_discrimination` is a training surrogate, **not exact PDS**.
The marginal and structure weights warm up over the first two epochs without
freezing a component. AdamW uses base learning rate 2e-4 and weight decay 1e-4.
The finite-cloud mean uses a documented straight-through derivative through the
rounded detection count; hard rank permutation uses the original differentiable
training surrogate. Forward inference emits the exact finite cloud.

## Validation and convergence

Validate every two epochs on four fixed panels:

| Panel | Response conditions | Distribution conditions | Weight |
|---|---:|---:|---:|
| External held context | 256 | 32 | 1/2 |
| Each of three internal held-condition contexts | 64 | 16 | 1/6 |
| Total | 448 | 80 | 1 |

All four panels' predictions are generated and sealed before their query labels
are opened. The selected objective is the fixed weighted sum of output losses;
the shared auxiliary loss does not select a checkpoint. No rank over candidates,
test score, historical performance target or PDS floor is used for selection.
Exact score ties select the earlier trained checkpoint.

The registered plateau controller requires at least 10 epochs and 1,500 optimizer
steps, six eligible checks without a substantive 0.2% relative objective gain,
learning-rate reductions at plateau checks 2 and 4, and at least two epochs after
the last reduction. The initial budget of 60 epochs extends to 120 if needed.
Reaching the hard cap without convergence yields `needs_decision`; it does not
open the formal test gate. Restarts restore optimizer, RNG, schedule position,
stopping state and current run inputs.

## Final prediction and aggregation

The selected checkpoint generates 512-cell clouds for all 1,525 test conditions
and all 5,979 genes. Means are the actual emitted cloud means. Every completed
prediction file and its checkpoint are sealed before the test labels are read.
No test-guided refitting, calibration or checkpoint reselection is performed.

For each metric, average the three initialization scores within each held
context. Average these five context means equally. Report their sample variance
and standard deviation (ddof=1), plus the within-context seed variability.
Missing/undefined values retain valid counts and nulls. The 15 runs are not 15
independent cell-context replicates. Aggregating an incomplete set must be marked
explicitly; it is not a five-context main-table result.
