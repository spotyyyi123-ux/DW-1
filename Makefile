# DW-1 Makefile: удобные обёртки. Работает и без make: смотри README.md
PY ?= python
PYTHONPATH := src

help: ## список команд
	@echo "DW-1: make venv | install | demo | test | lint | eval | attacks | report | data"

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
ifeq ($(ATTACKS_V2),1)
	PYTHONPATH=src $(PY) -m dw1.eval.attacks_v2 --predictions "$(ATTACK_PREDICTIONS)" --out "$(ATTACK_EVAL_OUT)"
else
ifeq ($(REVIEW_PACKAGE),)
ifeq ($(LLM),1)
	PYTHONPATH=src $(PY) scripts/evaluate_sprint3_llm.py
else
	$(if $(CAMPAIGNS),$(PY) -m dw1.eval.campaigns --campaigns $(CAMPAIGNS) --annotations $(ANNOTATIONS) --out data/runs/$(RUN),$(PY) -m dw1.eval.harness --run $(RUN))
endif
else
	$(PY) -m dw1.eval.campaign_review --package "$(REVIEW_PACKAGE)" --embeddings "$(EMBEDDINGS)" --annotations "$(ANNOTATIONS)" --reviewers $(REVIEWERS) $(if $(ADJUDICATED_BY),--adjudicated-by "$(ADJUDICATED_BY)") --out "$(EVAL_OUT)" --report "$(CAMPAIGN_REPORT)"
endif
endif

EVAL_OUT ?= data/runs/sprint3/campaigns/eval
CAMPAIGN_REPORT ?= docs/notes/sprint3-campaigns.md

attacks:
	$(PY) scripts/attacks.py --benchmark data/sprint2/bench --out data/runs/sprint2/attacks

ATTACK_BENCHMARK ?= data/sprint2/bench
ATTACK_PACKAGE ?= data/runs/sprint4/attacks/package
ATTACK_PREDICTIONS ?= data/runs/sprint4/attacks/scored/predictions.parquet
ATTACK_EVAL_OUT ?= data/runs/sprint4/attacks/eval
ATTACK_MODELS ?= data/runs/sprint4/attacks/models.json
ATTACK_PROVENANCE ?= data/runs/sprint4/attacks/provenance.json
ATTACK_SCORED_OUT ?= data/runs/sprint4/attacks/scored

attacks-v2-plan:
	$(PY) scripts/attacks_v2.py plan --benchmark "$(ATTACK_BENCHMARK)" --models "$(ATTACK_MODELS)"

attacks-v2-prepare:
	$(PY) scripts/attacks_v2.py prepare --benchmark "$(ATTACK_BENCHMARK)" --provenance "$(ATTACK_PROVENANCE)" --out "$(ATTACK_PACKAGE)"

attacks-v2-score:
	$(PY) scripts/attacks_v2.py score --package "$(ATTACK_PACKAGE)" --external "$(ATTACK_EXTERNAL)" --models "$(ATTACK_MODELS)" --semantic-review "$(ATTACK_SEMANTIC_REVIEW)" --out "$(ATTACK_SCORED_OUT)"

report:
	PYTHONPATH=src $(PY) -m dw1.eval.report --run $(RUN)

data:
	PYTHONPATH=src $(PY) scripts/fetch_datasets.py --plan

# Внешние LLM-скоры не вычисляются на локальном Mac: здесь только подготовка и калибровка.
llm-data:
	PYTHONPATH=src $(PY) scripts/prepare_llm_benchmark.py

llm-eval:
	PYTHONPATH=src $(PY) -m dw1.eval.llm_calibration

# Frozen real top-50 population; local commands do not import torch.
campaign-review:
	$(PY) -m dw1.campaigns.review --campaigns "$(CAMPAIGNS)" --documents "$(DOCUMENTS)" --provenance "$(PROVENANCE)" --out data/runs/sprint3/campaigns/review

campaign-eval:
	$(PY) -m dw1.eval.campaign_review --package data/runs/sprint3/campaigns/review --embeddings "$(EMBEDDINGS)" --annotations "$(ANNOTATIONS)" --reviewers $(REVIEWERS) $(if $(ADJUDICATED_BY),--adjudicated-by "$(ADJUDICATED_BY)") --out data/runs/sprint3/campaigns/eval --report docs/notes/sprint3-campaigns.md
