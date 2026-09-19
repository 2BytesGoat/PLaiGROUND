# PLaiGROUND — Long-Term Vision

> Working notes from the Sep 2026 ideation session. This is the north star,
> not a spec. Decisions may evolve; the reasoning is recorded so we don't
> re-litigate closed questions.

## What this project is

An open-source, visual (drag-and-drop) tool for teaching ML to absolute
beginners — kids, highschoolers, uni students, coders with no ML background —
by building AI that plays games. The flagship game is DragonJump
(dragon-jump-remaster); more games come later as "packs".

PLaiGROUND is the practical/lab counterpart of 2bytesgoat.com (theory lives
on the site, hands-on lives here). No lessons in-repo — just code. Inspired
by while True: learn (node-graph ML puzzles).

Long-term commercial idea: a Steam release of the tool. Code stays open
source regardless.

## The learning ladder (core design principle)

1. **Kids** — press RUN on a prebuilt mission, watch their agent play the
   game live, tweak one knob (epochs, "jumpiness").
2. **Highschoolers** — wire prebuilt blocks in the node editor and see how
   data flows.
3. **Coders** — open a block's source, copy it, write their own block.
4. **Uni students** — export the pipeline to clean PyTorch (ml_forge's
   File -> Export), leave the tool behind.

The magic moment is watching your agent get better at the game — the game
itself (Godot) is the primary visualization. UI visualization of the game
inside the node editor (Inspect block) is a high-value later feature.

## Architecture decisions

### Engine: fork of ML Forge (zaina-ml/ml_forge, MIT) — DECIDED

- Fork, own it, diverge freely; propose an upstream merge only if it still
  makes sense later. Code stays open source.
- Why ml_forge: MIT license (Steam/open-core friendly), small readable
  codebase (~25 files), Data Prep -> Model -> Training tab flow maps 1:1 to
  the teaching curriculum, block schema (label/params/inputs/outputs/
  defaults/tooltip/when_to_use) is a clean teaching artifact, and it already
  has export-to-PyTorch.
- The engine stays **game-agnostic**: graph editor, blocks registry,
  codegen, headless runner, GUI. Game-specific stuff lives in packs.
- Keep ml_forge's original vision domain (MNIST/CIFAR image classification)
  untouched — it still teaches CNNs.

### Why not ComfyUI — DECIDED (do not revisit without new facts)

- ComfyUI is **GPL-3.0**: viral license forecloses open-core (free OSS tool +
  paid Steam edition) and taints the whole app; ml_forge (MIT) keeps both
  options open.
- It's a diffusion engine wearing a node UI (10-20x larger codebase; surgery
  to adapt); web-server architecture (Python backend + browser frontend) is
  heavy for Steam packaging; one flat canvas vs ml_forge's curriculum-shaped
  tabs; "read the source to learn" is unrealistic there.
- ComfyUI remains a **UX donor**: custom_nodes folder convention for block
  sharing, node color coding, workflow templates, one-click manager pattern.

### Alternatives swept (Sep 2026) — DECIDED (keep ml_forge fork)

Full sweep of node-editor / no-code-ML candidates (license + activity
verified via GitHub API before deciding). Nothing beats the ml_forge fork
for this project's requirements: MIT, readable Python blocks students can
edit, curriculum structure, export-to-PyTorch, headless runner, small
codebase.

| Option | License | Status | Why not |
|---|---|---|---|
| Langflow (155k★) | MIT | active | LLM-agent workflow builder (API chains, chatbots) — wrong domain, huge codebase |
| Nodezator (2.8k★) | Unlicense | active | Strongest generic plan-B: permissive + maintained, but flat canvas (no curriculum), no codegen/export, no headless runner, arbitrary-function blocks — less pedagogically shaped |
| PyFlow (3.2k★) | Apache-2.0 | stale (Sep 2025) | Generalist visual scripting, dormant upstream |
| NodeGraphQt (1.8k★) | MIT | maintained | Qt *framework* for building editors, not an app — equals building from scratch |
| Ryven (4.1k★) | MIT | **archived** | Dead upstream — forking means adopting an abandoned editor |
| xyflow / React Flow (38.4k★) | MIT | very active | TypeScript UI library, not a Python execution engine — would mean writing backend/codegen/save format from scratch |
| ComfyUI (70k+★) | GPL-3.0 | very active | See "Why not ComfyUI" above |

- GitHub sweeps for "visual ML teaching" / "drag-and-drop AI training"
  found only 0-1★ hobby projects — **the niche is genuinely empty**, which
  strengthens the Steam-era moat.
- **Editor is swappable**: the `.mlf` schema and block registry are
  GUI-agnostic. If pygame ever proves limiting (Steam polish, classrooms,
  web embeds for 2bytesgoat.com), a custom editor on React Flow (MIT,
  very active) reading/writing the same `.mlf` is the designated escape
  hatch — Python blocks don't change, only the canvas. v2/Steam-era
  decision, portability costs nothing today.

### Packs — the game-agnostic layer — DECIDED

- Packs live in PLaiGROUND: `packs/dragonjump/` = observation schema, block
  groups, missions, eval protocol. A new game = a new pack; engine untouched.
- Pack includes an eval-protocol spec so leaderboards work identically for
  every game.

### Headless runner — DECIDED (build first, before GUI domain work)

- `ml-forge run pipeline.mlf` (no pygame): loads JSON project, validates,
  executes, prints metrics, exit code. Needed for CI, `just` targets,
  Docker, and the future Steam launcher. GUI is for teaching locally;
  headless is for automation.

## Curriculum roadmap (block families)

- **v1 — parity with today's scripts** (behavior cloning on DragonJump):
  - Sources: LiveEnv (godot_env), RecordedSessions (jsonl)
  - Policies: Random, HeuristicJump (the if/else wall-check)
  - Recording: KeyboardRecorder (SPACE=jump, R=discard, ESC=quit)
  - Features: ParseObservation/Grid, CanJump, Raycasts, WallDistances,
    ObjectOneHot, Sensors (lifted from src/processing/)
  - Models: DecisionTree, RandomForest, LogisticRegression (NOT plain linear
    regression — action is binary; decided), MLP (torch)
  - Training: SessionSplit (GroupShuffleSplit), ClassBalance, BCTrain
    (epochs/early-stop/checkpoint), LossPlot, Evaluate
  - Deploy: AgentRunner (checkpoint -> env loop), later Inspect block
- **v2 — teaching polish**: Inspect/visualizer block (renders grid frames,
  feature vectors, loss curves, tree diagrams inside the node graph — the
  while-True-learn feel), export-to-PyTorch polish, mission
  certificates/scores.
- **v3 — 9-years-of-ML-engineering topics**, roughly in order:
  genetic algorithms (evolve MLP weights — no gradients, very visual),
  reinforcement learning (sb3 wrapper already exists; reward design as
  gameplay), computer vision on game frames, signal processing.
- Game-visualizer-in-UI (render DragonJump inside the editor) — after the
  domain stabilizes.

## Missions (replace the numbered scripts)

- Templates as **missions with win conditions** ("Mission 3: get the tree to
  clear the first spike — target 30s survival"), verified by a seeded eval
  run. Mission index (missions.json or MISSIONS.md) lets 2bytesgoat.com link
  to and track progress.
- Equivalents of 00_random, 01_if_else, 02_record, 03_decision_tree,
  04_random_forest, 05_mlp (from playground.py).
- Old scripts are deleted after each template verifiably reproduces the
  behavior (accuracy parity on same data). Full replacement, not wrappers.

## Distribution & setup

- Student flow: get the repo -> `just setup` (uv) -> get the game (free
  demo from itch or Steam) -> launch -> play/train. Game is a black box
  (exported binary); students never see game code.
- **just** is the cross-OS task runner (Windows has no make): installed free
  via the pypi wheel (rust-just) that ships with uv. One justfile for all
  OSes; Makefile may remain as a thin alias.
- **Game discovery block**: if the game binary is missing -> "Do you have a
  Steam account? (y/n)" -> open the Steam page or the itch page. Never
  direct downloads (protects sales); free demos on both stores will all be
  training-compatible. Also handle "demo build too old — update on the
  store page" via protocol-version check.
- Docker: kept only so there's no macOS build burden; smooth targets are
  Linux and Windows. Docker is for reproducibility, not a student touchpoint.

### Dependency slimming — DECIDED direction

- torch (~2-3GB) is only needed for MLP/GA/RL missions -> optional uv extra
  (`uv sync --extra torch`); sklearn missions run without it (~300-500MB).
- CPU-only torch wheels by default; CUDA only on request.
- sb3 deferred to the RL missions.
- Future Steam build: bundled CPU torch, ~1.5-2GB, acceptable.

## Leaderboards (idea stage)

- Separate in-game leaderboard for AI models: humans, AIs, both.
- Fairness via deterministic seeded eval runs (fixed level, seed, action
  budget); anti-cheat (replay verification) deferred.

## Block sharing (deferred)

- Must exist eventually ("YES — we need to share blocks").
- Near term: drop a .py block into a blocks/ folder (ComfyUI-style).
- Long-term: Steam Workshop if/when the Steam app happens.

## Open questions (parked)

- Exact mission list + win conditions.
- Steam edition shape (single GPL-ish free app vs open-core; moot while the
  fork is MIT).
- Leaderboard protocol details & submission format.
- Fork hosting mechanics (GitHub fork + submodule was the working assumption).

## Related context

- Protocol: game's training mode speaks TCP on port 11008 (protocol v0.7,
  matching src/wrappers/godot_env.py); launch WITHOUT the `--` separator —
  see dragon-jump-remaster docs/technical/training.md.
- The old numbered scripts (src/00-04*.py, playground.py) remain the
  behavioral reference until mission parity is verified.