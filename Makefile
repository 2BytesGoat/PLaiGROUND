# PLaiGROUND — quick setup and run helpers.
# Requires: uv (https://docs.astral.sh/uv/) and make.
#
#   make setup-project   Install Python 3.13 + all dependencies into .venv
#   make setup-docker    Build the container image and run Jupyter Lab in it
#   make run-random      Run the random agent against the game in ./environments

.PHONY: setup-project setup-docker run-random

setup-project:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv is not installed. Install it with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	uv python install 3.13
	uv sync
	@echo ""
	@echo "Setup complete. Try it out with: make run-random"

setup-docker:
	@command -v docker >/dev/null 2>&1 || { \
		echo "docker is not installed. Get it from https://www.docker.com/products/docker-desktop/"; \
		exit 1; \
	}
	docker compose up --build

run-random:
	uv run python src/00_random_agent.py