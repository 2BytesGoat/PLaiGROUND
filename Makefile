# Convenience commands for the PLaiGROUND lab.

PYTHON ?= .venv/bin/python
UV ?= uv
EXPERIMENT ?= experiments/0001_ppo_clean_start_1_1

.PHONY: install test compile smoke-ppo help

install:
	$(UV) pip install --python $(PYTHON) -e . --no-deps

test:
	$(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q src/plaiground scripts tools

smoke-ppo:
	$(PYTHON) scripts/reinforcement/01_ppo.py --help

help:
	@echo "make install      install package editable into .venv"
	@echo "make test         run Python unit tests"
	@echo "make compile      byte-compile package and entry points"
	@echo "make smoke-ppo    show the PPO entry point help"
