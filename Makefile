# DW-1 Makefile: удобные обёртки. Работает и без make: смотри README.md
PY ?= python
PYTHONPATH := src

help: ## список команд
	@echo "DW-1: make venv | install | demo | test | lint | eval | report | data"

venv:
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -U pip

install:
	.venv/Scripts/pip install -e ".[dev,api,images]"

demo:
	PYTHONPATH=src $(PY) -m dw1 tests/fixtures/sample.eml

smoke:
	$(PY) scripts/smoke.py

test:
	PYTHONPATH=src $(PY) -m pytest

lint:
	PYTHONPATH=src $(PY) -m ruff check src tests
	PYTHONPATH=src $(PY) -m mypy src

eval:
	$(if $(CAMPAIGNS),$(PY) -m dw1.eval.campaigns --campaigns $(CAMPAIGNS) --annotations $(ANNOTATIONS) --out data/runs/$(RUN),$(PY) -m dw1.eval.harness --run $(RUN))

report:
	PYTHONPATH=src $(PY) -m dw1.eval.report --run $(RUN)

data:
	PYTHONPATH=src $(PY) scripts/fetch_datasets.py --plan
