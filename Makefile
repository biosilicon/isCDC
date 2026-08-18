PYTHON ?= python

.PHONY: setup test lint run import-example frontend-build frontend-test frontend-test-browser annotation-create annotation-test

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

frontend-build:
	cd frontend && npm run build

frontend-test:
	cd frontend && npm test

frontend-test-browser:
	cd frontend && npm run test:browser

annotation-create:
	conda env create -f annotation/environment.yml
	conda run -n iscdc-cell-annotation Rscript -e \
		'options(timeout=600); install.packages("https://cloud.r-project.org/src/contrib/renv_1.2.4.tar.gz", repos=NULL, type="source")'
	conda run -n iscdc-cell-annotation Rscript -e \
		'renv::restore(lockfile="annotation/renv.lock", library=.libPaths()[1], prompt=FALSE)'

annotation-test:
	conda run -n iscdc-cell-annotation env PYTHONPATH=src \
		python -m pytest annotation/tests tests/test_cell_type_annotation.py tests/test_cell_type_visualization.py
	conda run -n iscdc-cell-annotation Rscript annotation/test_single_r_contract.R
	conda run -n iscdc-cell-annotation Rscript annotation/test_census_reference_contract.R
