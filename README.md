# PLaiGROUND

PLaiGROUND (name may change) is a repository for learning Machine Learning by training Game AIs. 

## 🍽️ Table of Contents

Here are some nifty links to navigate this huge README file

- 👷 [Local Setup](#-local-setup) - What you need to do to setup the project bare-metal
- 🐋 [Container Setup](#-container-setup-docker--podman) - What you need to do to setup the project using Docker or Podman
- 🏃‍➡️ [Running the project](#️-running-the-project) - How to start your first random agent
- 📋 [What's to come](#-whats-to-come) - A short ToDo for people to know that's coming
- 🐐 [The GOATs](#-the-goats) - A shout out to all the people that helped along the way
- 🤙 [Contact](#-contact) - In case you have any issues or just want to chat


Choose either of these two ways of setting up your environment. No need to do both Container and Local setup.

> **Important:** Local setup is currently supported on **Linux/Windows only**.  
> On **macOS**, please use the Container setup.

## 👷 Local Setup

All you need is [uv](https://docs.astral.sh/uv/) and [make](https://www.gnu.org/software/make/). uv will automatically download Python 3.13 and install every dependency into a local `.venv`:

```
make setup-project
```

That's it. To try it out, place the game executable in `./environments` (see [Running the project](#️-running-the-project)) and run:

```
make run-random
```

### Available Make targets

| Target | What it does |
|--------|--------------|
| `make setup-project` | Installs Python 3.13 + all dependencies via uv |
| `make setup-docker` | Builds the container image (via `docker compose`) and starts Jupyter Lab in it |
| `make run-random` | Runs the random agent (`src/00_random_agent.py`) |

## 🐋 Container Setup (Docker / Podman)
### 1. Download and Install Docker
Docker is an application that's used to make sure you don't have OS compatibility issues when setting up environments. You can download [Docker Desktop](https://www.docker.com/) from their official website. (Podman with a docker alias works just as well.)

### 2. Build and run the container

Once Docker is installed, one command does everything:

```
make setup-docker
```

This runs `docker compose up --build` against the included `compose.yaml`, which:
1. Builds a container image with all necessary dependencies (installed via uv from `uv.lock`)
2. Starts a container with Jupyter Lab running inside it
3. Maps port 8888 on your machine to the container's Jupyter server

The container will continue running until you stop it with Ctrl+C in the terminal where you ran the command.

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
* Test the setup by running the random_agent script
```
make run-random
```

#### 🐋 With Containers
* Download the **Linux version** of 👉 [Dragon Jump from itch.io](http://2bytesgoat.itch.io/dragon-jump)
* Unzip the game executables inside the `./environments` folder
* Test the setup by running the random_agent notebook
```
http://127.0.0.1:8888/lab/tree/notebooks/00_random_agent.ipynb
```
You can run individual code snippets by pressing `SHIFT + ENTER`

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

