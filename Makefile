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
ARGS ?=
T ?=

# repo-brief owns index/context/preflight/graph/archive/writing/skill. The
# fragment is included at the bottom so its targets win; set REPO_BRIEF to a
# checkout to develop against it instead of the submodule.
REPO_BRIEF ?= tools/vendor/repo-brief

.PHONY: install test compile smoke-ppo brief run search level-data level-status watch help

install:
	$(UV) pip install --python $(PYTHON) -e . --no-deps

test:
	$(PYTHON) -m unittest $(if $(T),$(T),discover -s tests)

compile:
	$(PYTHON) -m compileall -q src/plaiground scripts tools

smoke-ppo:
	$(PYTHON) scripts/reinforcement/01_ppo.py --help

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
	@echo "make test T=tests/test_x.py  run one test file"
	@echo "make compile      byte-compile package and entry points"
	@echo "make smoke-ppo    show the PPO entry point help"
	@echo "make index        regenerate the committed .repo-brief/map.md"
	@echo "make index-check  fail if the map's structure is stale"
	@echo 'make context q="..."  rank code, experiments, docs and decisions'
	@echo 'make context ARGS=--brief q="..."  one-screen start'
	@echo "make graph        dependency view; graph <area> or ARGS='--task <query>'"
	@echo "make preflight    map, profile, front doors, dates, references"
	@echo "make archive      review finished docs; ARGS='--apply <path>'"
	@echo "make writing      write docs/meta/writing.md from detection"
	@echo "make skill        install or verify the document skill"
	@echo "make brief        repo state: branch, game, running jobs, ledger, recent runs"
	@echo "make run RUN=...  one run's checkpoints, evals, and log tail"
	@echo 'make search q="..." [SEARCH_DIR=.] [SEARCH_GLOB=*.py]  bounded search'
	@echo "make level-status LEVEL=1-9  show level experiments and data summaries"
	@echo "make level-data LEVEL=1-9 LEVEL_RUN=runs/... LEVEL_GEN=46  build tiers/pools"
	@echo "make watch        play the 1.555M 1-1..1-4 chain in the game window"
	@echo "                  WATCH_LEVEL=1-4 WATCH_SPEEDUP=2 WATCH_MODEL=<checkpoint>"

# Included last: the fragment owns index/context/preflight/graph/archive/
# writing/skill, so its definitions must come after anything in this file.
-include $(REPO_BRIEF)/make/repo-brief.mk