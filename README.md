# PLaiGROUND

PLaiGROUND (name may change) is a repository for learning Machine Learning by training Game AIs. 

> **Lab state (2026-10-27):** historical experiment scripts, runs, and notes
> were moved to
> [`archive/2026-10-27-cleanup/`](archive/2026-10-27-cleanup/README.md).
> The active branch is the 794-value sensor PPO pipeline described there.
> The model actually installed in Godot is documented in
> [`docs/deployed-model-58.md`](docs/deployed-model-58.md).
> Sections below include historical experiments and are not the current plan.

## 🍽️ Table of Contents

Here are some nifty links to navigate this huge README file

- 👷 [Local Setup](#-local-setup) - What you need to do to setup the project bare-metal
- 🐋 [Docker Setup](#-docker-setup) - What you need to do to setup the project using Docker
- 🏃‍➡️ [Running the project](#️-running-the-project) - How to start your first random agent
- 📋 [What's to come](#-whats-to-come) - A short ToDo for people to know that's coming
- 🐐 [The GOATs](#-the-goats) - A shout out to all the people that helped along the way
- 🤙 [Contact](#-contact) - In case you have any issues or just want to chat


Choose either of these two ways of setting up your environment. No need to do both Docker and Local setup.

> **Important:** Local setup is currently supported on **Linux/Windows only**.  
> On **macOS**, please use the Docker setup.

## 👷 Local Setup
### 1. Supported operating systems
Local setup is currently supported on:
- Linux
- Windows

For macOS, use the Docker setup below.

### 2. Download and Install Python
This project uses **[Python 3.13](https://www.python.org/downloads/release/python-3131/)**.

### 3. Create a virtual environment

Create a local virtual environment in the project folder:
```
python -m venv .venv
```

### 4. Activate the virtual environment

Depending on your OS:
```
# On Windows
.\.venv\Scripts\activate

# On Linux
source .venv/bin/activate
```

### 5. Install Poetry inside the virtual environment

With the virtual environment activated:
```
python -m pip install poetry
```

### 6. Install dependencies with Poetry

Run:
```
poetry install
```

### ⚠️ Troubleshooting (Windows)

**Can't activate environment on Windows**

If you see an error similar to this
```
At line:1 char:1
+ .\.venv\Scripts\activate
+ ~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```
You’re hitting a PowerShell execution policy restriction—Windows is blocking the virtual environment activation script (Activate.ps1) from running.

Run PowerShell as your user (no admin needed) and execute:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then confirm with Y, and try again:
```
.\.venv\Scripts\activate
```

**Can't install the numpy library because of missing compilers?**

Go to the official Visual Studio download page:
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

Install the Build Tools:
* Launch the installer.
* Select:
    * "C++ build tools"
* In the right panel, make sure the following are selected:
    * MSVC v14.x (latest version)
    * Windows 10/11 SDK
    * CMake tools for Windows
    * C++ CMake tools for Windows

You may also want to try:
```
python -m pip install meson ninja cython pybind11
```

## 🐋 Docker Setup
### 1. Download and Install Docker
Docker is an application that's used to make sure you don't have OS compatibility issues when setting up environments. You can download [Docker Desktop](https://www.docker.com/) from their official website.

### 2. Creating a Docker Image
Once Docker is installed, you can use our Docker script that will run all the commands for you:
```
# make the script executable
chmod +x run_docker.sh

# run the script that does the docker setup for you
./run_docker.sh
```

After running this script, Docker will:
1. Build a Docker image with all necessary dependencies
2. Start a container with Jupyter Lab running inside it
3. Map port 8888 on your machine to the container's Jupyter server

The container will continue running until you stop it with Ctrl+C in the terminal where you ran the script.

### 3. Access Jupyter Lab
If everything went well, you should be able to access the plAIground by navigating to:
```
http://127.0.0.1:8888/lab?token=plaiground
```

Here you can browse the notebooks, run code, and interact with the game environments.

## 🏃‍➡️ Running the project

### 👷‍♂️ With Local Setup
* Download 👉 [Dragon Jump from itch.io](http://2bytesgoat.itch.io/dragon-jump)
* Place the game executable inside the `./environments` folder
* Activate your virtual environment
```
# On Windows
.\.venv\Scripts\activate

# On Linux
source .venv/bin/activate
```
* Test the setup by running the random_agent script
```
python src/00_random_agent.py
```

#### 🐋 With Docker
* Download the **Linux version** of 👉 [Dragon Jump from itch.io](http://2bytesgoat.itch.io/dragon-jump)
* Unzip the game executables inside the `./environments` folder
* Test the setup by running the random_agent notebook
```
http://127.0.0.1:8888/lab/tree/notebooks/00_random_agent.ipynb
```
You can run individual code snippets by pressing `SHIFT + ENTER`

## 🧬 NEAT experiment

The NEAT experiments evaluate a whole population in the existing multi-agent
training environment and use the game's reward directly as fitness. The
lineage runner carries one population through the campaign in order; the
per-level batch runner is still available as an independent baseline. See
[docs/neat-1-1.md](docs/neat-1-1.md) and
[docs/neat-lineage.md](docs/neat-lineage.md).

## 🤖 PPO pilot

The PPO experiments train one shared policy through standard Gymnasium
wrappers. The single-level pilot uses isolated Godot training worlds per
parallel worker; the campaign lineage carries the same model through 1-1 to
1-16 one level at a time, with zero-shot and post-stage evaluations. Gradient
updates reuse transitions instead of spending a whole population evaluation
per generation. See [docs/ppo-pilot.md](docs/ppo-pilot.md).

The distillation experiment merges the per-level PPO snapshots into one
level-agnostic student using supervised learning. The selected student sees the
58 game observations only — no campaign-level ID — and clears **all 16 campaign
levels at 100/100 agents**, choosing its own first jump. Its verified JSON
export is bundled at `godot/src/resources/ai/student_obs58_256.json` and runs
in-game through the pure-GDScript `MLPPolicy` runtime (all 16 levels finish
headlessly at 8×). See [docs/ppo-distillation.md](docs/ppo-distillation.md).

## 📈 Model improvement experiments (historical; archived)

The current student finishes the campaign but is route-slow on levels such as
1-7, 1-10, and 1-11. The first pass retrained all 16 teachers with the
training-only time reward profile and eval patience, selected per-level
champions, then distilled them into the global student. Champion selection uses
speed first, with efficiency/flow tie-breaks; multi-seed PPO restarts are
available after a plateau. A live-takeover script tests whether an improved
teacher can recover from student-visited states. The new multi-hot observation
(`--obs-encoding multi_hot`, 754 values) is implemented. The first multi-hot
experiment was negative: a fresh 1-11 legacy 58/64×64 PPO reached **3.8667 s**,
while multi-hot 64×64 stayed at 0/100, multi-hot 256×256 reached 4.4667 s
after 775k steps, and multi-hot 512×512 failed at every eval (0/100; one early
stochastic finish, then no more). The all-16 multi-hot teacher retrain was not started. The
256-wide legacy-teacher student cleared 13/16, failing 1-11/1-14/1-15. The
bundled JSON remains the legacy 58-value model. The multi-hot teacher driver and spec-2 export path are in the cleanup archive.

Selected result: the new DAgger student clears all 16 levels at 100/100 with a
**78.58 s** campaign best-time sum (old: 83.12 s). The 1-11 teacher improved from
9.53 s to 8.25 s in the full sweep; the 8.05 s smoke teacher was tried again but
regressed 1-3 and was rejected. Recovery tests finished 100/100 on 1-8 and 1-11.
NEAT, architecture search, and dropout are deferred to V2. See
[archive/2026-10-27-cleanup/docs/model-improvement-experiments.md](archive/2026-10-27-cleanup/docs/model-improvement-experiments.md).

## 📋 What's to come:

- [ ] add notebook for how to build an If-Else agent
- [ ] add notebook for how to build a Decision Tree

## 🐐 The GOATs

This repo is basically an amalgamation of information I gathered from multiple smart and passionate people. Without them making their work publicly available for free, this repo wouldn't exist.

* **[GodotRL](https://github.com/edbeeching/godot_rl_agents)** - I would never have started working on this repo if it weren't for **Ed Beeching** and his wonderful Godot library for training Machine Learning agents.
* **[Heartbeast](https://www.youtube.com/@uheartbeast)** - I probably would not know Godot if it weren't for **Benjamin** and his Godot tutorials. He is an awesome teacher that lit the fire inside my heart, both for learning Godot and teaching others.
* **[Sentdex](https://www.youtube.com/@sentdex)** - The first ever video I watched about computer vision and machine learning was from **Harrison Kinsley**. On my first internship I used his tutorials to learn about Haar Cascades and Image classification. If it wasn't for him, I wouldn't be a Machine Learning engineer today.

And for this, I thank you from the bottom of my heart 🙇‍♂️

## 🤙 Contact

For questions regarding this repo and environments 
* 2BytesGoat - [Discord](https://discord.gg/FsKQPupcVs)

For questions regarding godot-rl
* Godot RL Agents - [Github](https://github.com/edbeeching/godot_rl_agents), [Discord](https://discord.gg/HMMD2J8SxY)

