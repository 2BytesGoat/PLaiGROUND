# Convenience commands for the PLaiGROUND lab.

PYTHON ?= .venv/bin/python
UV ?= uv
EXPERIMENT ?= experiments/0001_ppo_clean_start_1_1

# `make watch` plays the selected 1-4 chain checkpoint in the game window.
WATCH_MODEL ?= runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
WATCH_CONFIG ?= experiments/0079_show_chain_1_1_to_1_4_0073
WATCH_LEVEL ?=
WATCH_SPEEDUP ?=
WATCH_TAG ?= watch

LEVEL ?= 1-9
LEVEL_RUN ?=
LEVEL_GEN ?=
LEVEL_ANALYSIS ?= ../dragon-jump-remaster/runs/level_analysis/$(LEVEL).json
LEVEL_TAG ?=
LEVEL_OUTPUT_ROOT ?=
LEVEL_MIN_ROUTES ?= 25
LEVEL_OPTIONAL_ARGS = $(if $(LEVEL_OUTPUT_ROOT),--output-root $(LEVEL_OUTPUT_ROOT)) $(if $(LEVEL_TAG),--tag $(LEVEL_TAG))

SEARCH_DIR ?= .
SEARCH_GLOB ?=
SEARCH_ALL ?=
RUN ?=

.PHONY: install test compile smoke-ppo index index-check context brief run search level-data level-status watch help

install:
	$(UV) pip install --python $(PYTHON) -e . --no-deps

test:
	$(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q src/plaiground scripts tools

smoke-ppo:
	$(PYTHON) scripts/reinforcement/01_ppo.py --help

index:
	$(PYTHON) tools/repo_index.py

index-check:
	$(PYTHON) tools/repo_index.py --check

context:
	@if [ -z "$(q)" ]; then echo 'usage: make context q="<task>"'; exit 2; fi
	$(PYTHON) tools/context.py "$(q)"

brief:
	$(PYTHON) tools/brief.py

run:
	@if [ -z "$(RUN)" ]; then echo 'usage: make run RUN=<name>'; exit 2; fi
	$(PYTHON) tools/brief.py --run "$(RUN)"

search:
	@if [ -z "$(q)" ]; then echo 'usage: make search q="<text>" [SEARCH_DIR=.] [SEARCH_GLOB=*.py]'; exit 2; fi
	$(PYTHON) tools/search.py "$(q)" --dir "$(SEARCH_DIR)" $(if $(SEARCH_GLOB),--glob "$(SEARCH_GLOB)",) $(if $(SEARCH_ALL),--all,)

level-status:
	$(PYTHON) tools/level.py status $(LEVEL)

level-data:
	@if [ -z "$(LEVEL_RUN)" ]; then echo 'usage: make level-data LEVEL=1-9 LEVEL_RUN=runs/<neat-run> LEVEL_GEN=<generation>'; exit 2; fi
	$(PYTHON) tools/level.py data \
		--level $(LEVEL) \
		--run-dir $(LEVEL_RUN) \
		$(if $(LEVEL_GEN),--generation $(LEVEL_GEN),) \
		--analysis $(LEVEL_ANALYSIS) \
		--min-routes $(LEVEL_MIN_ROUTES) \
		--seed 0 $(LEVEL_OPTIONAL_ARGS)

watch:
	@echo "watching $(WATCH_MODEL)"
	@echo "  levels: $(if $(WATCH_LEVEL),$(WATCH_LEVEL) only,from $(WATCH_CONFIG))  speedup: $(if $(WATCH_SPEEDUP),$(WATCH_SPEEDUP),from config)"
	$(PYTHON) tools/evaluate.py --config $(WATCH_CONFIG) --model $(WATCH_MODEL) --tag $(WATCH_TAG) --show $(if $(WATCH_LEVEL),--level $(WATCH_LEVEL)) $(if $(WATCH_SPEEDUP),--speedup $(WATCH_SPEEDUP))

help:
	@echo "make install      install package editable into .venv"
	@echo "make test         run Python unit tests"
	@echo "make compile      byte-compile package and entry points"
	@echo "make smoke-ppo    show the PPO entry point help"
	@echo "make index        regenerate docs/code-map.md and .cache/repo-index.json"
	@echo "make index-check  fail if docs/code-map.md is stale"
	@echo 'make context q="..."  search code, experiments, docs, and runs'
	@echo "make brief        repo state: branch, game, running jobs, ledger, recent runs"
	@echo "make run RUN=...  one run's checkpoints, evals, and log tail"
	@echo 'make search q="..." [SEARCH_DIR=.] [SEARCH_GLOB=*.py]  bounded search'
	@echo "make level-status LEVEL=1-9  show level experiments and data summaries"
	@echo "make level-data LEVEL=1-9 LEVEL_RUN=runs/... LEVEL_GEN=46  build tiers/pools"
	@echo "make watch        play the 1.555M 1-1..1-4 chain in the game window"
	@echo "                  WATCH_LEVEL=1-4 WATCH_SPEEDUP=2 WATCH_MODEL=<checkpoint>"
