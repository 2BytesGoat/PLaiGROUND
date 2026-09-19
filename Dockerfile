# Base image with uv preinstalled on Python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

# System deps. unzip is used by `just export-game` workflows; curl for
# healthchecks. No build-essential — every dependency ships a wheel.
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# The project venv lives OUTSIDE /app so runtime bind mounts can never
# shadow it with the host's macOS venv (uv then treats it as invalid and
# re-syncs gigabytes of wheels through the slow macOS file-sharing layer).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ---------------------------------------------------------------------------
# lean target (default): runs the Godot game + frame stream server only.
# No torch, no CUDA, no jupyter, no scikit-learn. Installs just numpy,
# gymnasium (wrappers) and python-dotenv (.config loading), and puts the
# ml_forge fork on PYTHONPATH instead of installing it (its UI deps
# dearpygui/Pillow are not needed for streaming).
#
# The fork is NOT baked in (no COPY --from=forge-src): it is bind-mounted
# at runtime by compose. This avoids the BuildKit-only `additional_contexts`
# feature, which podman's compose provider may not support.
# ---------------------------------------------------------------------------
FROM base AS lean

# The exported game build is x86_64 while Apple Silicon VMs are aarch64:
# the VM's qemu-x86_64 binfmt handler executes the game transparently, but
# it needs the x86_64 dynamic loader (/lib64/ld-linux-x86-64.so.2) and its
# glibc inside the container. Install the amd64 libc for that.
RUN dpkg --add-architecture amd64 \
    && apt-get update -qq \
    && apt-get install -y -qq libc6:amd64 \
    && rm -rf /var/lib/apt/lists/*

# Minimal runtime for wrappers/godot_env.py + src/stream_server.py
RUN uv pip install --system --no-cache numpy gymnasium python-dotenv

ENV PYTHONPATH="/ml_forge_plaiground"

# Keep the container alive; the stream server is started on demand via
# `docker compose exec -T` (see compose.yaml / ml_forge Docker Build node).
CMD ["sleep", "infinity"]

# ---------------------------------------------------------------------------
# full target (opt-in via `--profile full`): Jupyter Lab + the complete
# dependency set from uv.lock (torch, scikit-learn, opencv, matplotlib ...).
# ---------------------------------------------------------------------------
FROM base AS full

# Copy dependency definitions first for better layer caching.
# (uv.lock references the ml_forge fork as an editable path dep, so its
# project files must exist at /ml_forge_plaiground before uv sync runs)
COPY pyproject.toml uv.lock ./
COPY --from=forge-src / /ml_forge_plaiground

# Install all dependencies (including dev group: jupyter stack) from the lockfile
RUN uv sync --frozen --no-install-project

# Expose Jupyter notebook port + Godot RL socket port
EXPOSE 8888 11008

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=plaiground

# Run Jupyter Lab from the uv-managed environment
CMD ["uv", "run", "--frozen", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]