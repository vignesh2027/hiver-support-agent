-include .env
export

PY ?= .venv/bin/python
export PYTHONPATH := src
BRAND ?= GWRHelp

.DEFAULT_GOAL := help
.PHONY: help setup data scorecard episodes taxonomy index baselines pool prelabel label \
        classify generate judge grading-set grade report reproduce all test lint clean cost verify

help:
	@echo "Reproduce the headline results (no API key needed, uses committed cache):"
	@echo "  make setup       create .venv and install pinned deps"
	@echo "  make reproduce   rebuild every reported number from the cache   (~3 min)"
	@echo ""
	@echo "Full rebuild from raw data (needs GROQ_API_KEY, several hours of rate-limited calls):"
	@echo "  make data        download the 493 MB TWCS corpus"
	@echo "  make scorecard   score every brand on how learnable its support is"
	@echo "  make episodes    build GWRHelp episodes + chronological splits"
	@echo "  make taxonomy    cluster -> name -> consolidate the intent taxonomy"
	@echo "  make index       build the precedent retrieval index"
	@echo "  make pool        sample the 220-example golden pool"
	@echo "  make prelabel    two independent LLM pre-label passes"
	@echo "  make label       human adjudication CLI (interactive)"
	@echo "  make classify generate judge report"
	@echo ""
	@echo "  make test        run the test suite"
	@echo "  make cost        show LLM calls, tokens and throttle time actually spent"

setup:
	uv venv --python 3.11 .venv 2>/dev/null || python3 -m venv .venv
	$(PY) -m pip install -q -U pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "done. cp .env.example .env and add GROQ_API_KEY if you plan to rebuild."

data:
	@mkdir -p data/raw
	@test -f data/raw/twcs.csv || curl -L --progress-bar -o data/raw/twcs.csv \
	  "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
	@ls -lh data/raw/twcs.csv

scorecard: data
	$(PY) -W ignore -m hsa.data.prepare --scorecard

episodes: data
	$(PY) -W ignore -m hsa.data.prepare --brand $(BRAND)

taxonomy:
	$(PY) -W ignore -m hsa.taxonomy.induce --stage all --k 32
	$(PY) -W ignore -m hsa.taxonomy.schema

index:
	$(PY) -W ignore -m hsa.retrieve.index --backend tfidf --demo

baselines:
	$(PY) -W ignore -m hsa.classify.baselines

pool:
	$(PY) -W ignore -m hsa.evalx.sample_golden

prelabel:
	$(PY) -W ignore -m hsa.evalx.prelabel

label:
	$(PY) -W ignore -m hsa.tools.label_cli --mode intent

label-stats:
	$(PY) -W ignore -m hsa.tools.label_cli --stats

classify:
	$(PY) -W ignore -m hsa.evalx.harness --stage classify

generate:
	$(PY) -W ignore -m hsa.evalx.harness --stage generate

judge:
	$(PY) -W ignore -m hsa.evalx.harness --stage judge

grading-set:
	$(PY) -W ignore -m hsa.evalx.harness --stage grading-set

# Blind human grading of replies -> Cohen's kappa against the LLM judge.
grade:
	$(PY) -W ignore -m hsa.tools.label_cli --mode reply --limit 60

report:
	$(PY) -W ignore -m hsa.evalx.harness --stage report
	$(PY) -W ignore -m hsa.evalx.render_report

# The reviewer's entry point. HSA_OFFLINE=1 makes every LLM call a cache read,
# so this needs no API key, no network, and no rate-limit patience.
reproduce:
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.harness --stage classify
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.harness --stage generate
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.harness --stage judge
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.harness --stage report
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.render_report
	@echo ""
	@echo "Headline numbers are in artifacts/metrics/results.json and reports/RESULTS.md"

all: episodes taxonomy index pool prelabel classify generate judge report

test:
	$(PY) -m pytest tests/ -q

verify: test
	HSA_OFFLINE=1 $(PY) -W ignore -m hsa.evalx.harness --stage report >/dev/null && \
	  echo "offline replay OK"

cost:
	@$(PY) -c "import json;from hsa.llm import ledger_summary,cached_count;s=ledger_summary();\
print(json.dumps(s,indent=2));print('cached responses on disk:',cached_count())"

clean:
	rm -rf artifacts/runs artifacts/metrics/*.json reports/RESULTS.md
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
