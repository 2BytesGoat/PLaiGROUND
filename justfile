# PLaiGROUND — quick setup and run helpers.
# Requires: uv (https://docs.astral.sh/uv/)
# Run `uv run just` in this directory to list recipes (just ships as a project dep).
#
#   just setup-project          Install Python 3.13 + all dependencies into .venv
#   just setup-docker           Build the container image and run Jupyter Lab in it
#   just run-random             Run the random agent (game binary or Godot project, per src/.config)
#   just run-random-host        Run the random agent windowed on the host (Godot project launch)
#   just export-game            Export the Linux game build into environments/ (for Docker)
#   just docker-run-random      Run the random agent inside the container
#   just docker-forge-random    Run the ml_forge random-agent play loop inside the container
#   just forge                  Launch the ML Forge visual pipeline editor
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

# Run the random agent windowed on the host (launches the game from its Godot project)
run-random-host nb_agents="1" level="1-3":
    GODOT_PATH=/Applications/Godot.app GODOT_PROJECT_PATH=../dragon-jump-remaster SHOW_WINDOW=TRUE NB_AGENTS={{nb_agents}} LEVEL={{level}} uv run python src/00_random_agent.py

# Create the gitignored runtime_secrets.gd from the committed template if
# absent (mirrors the CI "Inject runtime secrets" step with dev defaults).
# Never clobbers a real secrets file that already exists.
bootstrap-secrets:
    test -f ../dragon-jump-remaster/src/scripts/singletons/runtime_secrets.gd \
        || cp ../dragon-jump-remaster/src/scripts/singletons/runtime_secrets.gd.template \
              ../dragon-jump-remaster/src/scripts/singletons/runtime_secrets.gd

# Export the Linux game build into environments/ (used by the Docker container)
export-game: bootstrap-secrets
    /Applications/Godot.app/Contents/MacOS/Godot --headless --path ../dragon-jump-remaster --export-release "Linux Desktop" ../PLaiGROUND/environments/DragonJump.x86_64

# Run the random agent inside the container (see setup-docker)
docker-run-random nb_agents="8" level="1-3":
    NB_AGENTS={{nb_agents}} LEVEL={{level}} docker compose exec plaiground uv run python src/00_random_agent.py

# Run the ml_forge random-agent play loop inside the container (same path the UI RUN button uses)
docker-forge-random steps="300":
    MAX_STEPS={{steps}} docker compose exec plaiground uv run python -c "from ml_forge.engine.game_run import _play_thread; _play_thread({'mode': 'random', 'max_steps': int('{{steps}}'), 'fps_limit': 0})"

# Launch the ML Forge visual pipeline editor (editable install of ../ml_forge_plaiground)
forge:
    uv run python -m ml_forge