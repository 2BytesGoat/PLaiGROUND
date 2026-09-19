# PLaiGROUND — quick setup and run helpers.
# Requires: uv (https://docs.astral.sh/uv/)
# Run `uv run just` in this directory to list recipes (just ships as a project dep).
#
#   just setup-project          Install Python 3.13 + all dependencies into .venv
#   just setup-docker           Build the container image and run Jupyter Lab in it
#   just run-random             Run the random agent (game binary or Godot project, per src/.config)
#
# run-random overrides:
#   just run-random 8 1-1

default:
    @just --list

# Install Python 3.13 + all dependencies via uv
setup-project:
    uv python install 3.13
    uv sync
    @echo ""
    @echo "Setup complete. Try it out with: just run-random"

# Build the container image (via docker compose) and start Jupyter Lab in it
setup-docker:
    docker compose up --build

# Run the random agent (src/00_random_agent.py)
run-random nb_agents="8" level="1-3":
    NB_AGENTS={{nb_agents}} LEVEL={{level}} uv run python src/00_random_agent.py