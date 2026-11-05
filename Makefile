PYTHON ?= python
DATA ?= data/processed
RAW ?= data/raw
WORK ?= work/j0
SPLITS ?= work/splits
PRIORS ?= data/priors/prior.npz
GRAPH ?= data/string/graphs/KS/A.npz
GRAPH_AXIS ?= data/string/source/node_axis.tsv
DEVICE ?= cuda:0
FOLD ?= F1
SEED ?= 17

.PHONY: help test smoke download download-plan prepare splits train benchmark export-native anonymity
help:
	@echo "SCULPT: test | smoke | download-plan | download | prepare | splits | train | benchmark | export-native | anonymity"
	@echo "Paths and device are overridable, e.g. make train DEVICE=cuda:0 FOLD=F1 SEED=17"

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider tests Compared_methods/tests

smoke:
	$(PYTHON) -m cosit smoke-test --output work/smoke

download-plan:
	$(PYTHON) scripts/download_data.py --raw-root $(RAW) --dry-run

download:
	$(PYTHON) scripts/download_data.py --raw-root $(RAW)

prepare:
	$(PYTHON) -m cosit prepare --config configs/datasets.yaml --raw-root $(RAW) --output $(DATA)
	$(PYTHON) -m cosit audit-data --data $(DATA) --axes configs/axes

splits:
	$(PYTHON) -m cosit split --data $(DATA) --config configs/j0.json --output $(SPLITS) --verify-frozen configs/splits

train:
	$(PYTHON) -m cosit train --data $(DATA) --priors $(PRIORS) --graph $(GRAPH) --graph-axis $(GRAPH_AXIS) --splits $(SPLITS) --fold $(FOLD) --seed $(SEED) --config configs/j0.json --output $(WORK)/$(FOLD)/seed_$(SEED) --device $(DEVICE)

benchmark:
	$(PYTHON) scripts/run_benchmark.py --data $(DATA) --priors $(PRIORS) --graph $(GRAPH) --graph-axis $(GRAPH_AXIS) --config configs/j0.json --output $(WORK) --device $(DEVICE)

export-native:
	$(PYTHON) -m Compared_methods.export_training --data $(DATA) --splits $(SPLITS) --fold $(FOLD) --output work/native/$(FOLD)/train.h5ad

anonymity:
	$(PYTHON) scripts/check_anonymity.py
