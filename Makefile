PYTHON ?= python

.PHONY: setup test lint run import-example

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

run:
	$(PYTHON) -m uvicorn iscdc.app:app --app-dir src --reload

import-example:
	PYTHONPATH=src $(PYTHON) -m iscdc.cli import-dataset \
		xenium_human_rcc_ffpe_rna_protein.h5mu \
		assets/examples/xenium_human_rcc_ffpe_rna_protein.metadata.yaml
