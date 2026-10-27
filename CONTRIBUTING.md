# Contributing

Install the development dependencies and run:

```bash
python -m pytest -q tests Compared_methods/tests
```

Keep the following invariants:

- Query perturbation labels never enter inference inputs.
- Internal held conditions are excluded from every gradient role.
- Fit scales, PCA and connection geometry using the registered training scope.
- Initialize every task-specific model from scratch per fold and seed.
- Do not add old predictor outputs or fixed source means as prediction anchors.
- Preserve gene/target axes, A2048 units, finite-cloud mean and rank marginals.
- Report actual results, missing values, uncertainty and any protocol changes.
- Do not commit data, downloaded upstream code, credentials, weights or results.

New comparisons must identify their native inputs, official commit, selection
rule, distribution semantics and any benchmark adaptation. A mean-only method
does not become a distribution method by adding arbitrary noise.
