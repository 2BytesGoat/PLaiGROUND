# PLaiGROUND

PLaiGROUND (name may change) is a repository for learning Machine Learning by training Game AIs. 

## 🍽️ Table of Contents

Here are some nifty links to navigate this huge README file

- 🐋 [Docker Setup](#-docker-setup) - What you need to do to setup the project using Docker
- 👷 [Local Setup](#-local-setup) - What you need to do to setup the project bare-metal
- 🏃‍➡️ [Running the project](#️-running-the-project) - How to start your first random agent
- 📋 [What's to come](#-whats-to-come) - A short ToDo for people to know that's coming
- 🐐 [The GOATs](#-the-goats) - A shout out to all the people that helped along the way
- 🤙 [Contact](#-contact) - In case you have any issues or just want to chat


Choose either of these two ways of setting up your environment. No need to do both Docker and Local setup.

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

## 👷 Local Setup
### 1. Download and Install Python
This project uses **[Python 3.13](https://www.python.org/downloads/release/python-3131/)**. You can try an older version of Python, but you may have compatibility issues.

### 2. Create a virtual environment 

By default Python will install a project's dependencies (requirements) globally. This may cause compatibility issues if you have multiple Python projects on your PC. In order to avoid this, you will need to use a virtual environment that will store a project's dependencies separately.
```
# Install the virtualenv library globally
python3 -m pip install venv

# Create a virtual environment inside the current folder
python3 -m venv .venv
```

### ⚠️ Troubleshooting

**On Windows** - Can't install the numpy library because of of missing compilers.

Go to the official Visual Studio download page: 
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/

Install the Build Tools
* Launch the installer.
* Select:
    * "C++ build tools"
* In the right panel, make sure the following are selected:
    * MSVC v14.x (latest version)
    * Windows 10/11 SDK
    * CMake tools for Windows
    * C++ CMake tools for Windows


### 3. Activate the virtual environment

Depending on your operating system, you will need to run a specific command to step into the virtual environment.
```
# On Windows
./.venv/Scripts/activate

# On Linux and MacOS
source ./.venv/bin/activate
```
You'll know the command worked if the command line you'll see `(.venv)` at the beginning of the line.

### 4. Install dependencies

The project has several dependencies that were stored inside the `requirements.txt` file which are needed to run the code.

```
# Install the dependencies inside the virtual environment
pip install --no-cache-dir -r requirements.txt
```

## 🏃‍➡️ Running the project

### 1. Downloading the game
Go to Steam and request access to the Open Playtest here:
* 👉 [Dragon Jump](https://store.steampowered.com/app/2471710/Dragon_Jump/) - One-button input precision platformer - similar to SuperMeatBoy

Once you've downloaded the game, you'll need to boot it up and open the developer console using the `~` (tilda) key and type in the following command:
```
learning on
```

This command enables the AI interface in the game, which allows external programs (like our agents) to control the game character. The AI Settings menu will appear whenever you select a level, allowing you to configure how the AI interacts with the game and monitor its performance.

### 2. Running the random agent script
A random agent is the most basic form of AI. 

#### 2.1. 🐋 With Docker
You'll need to open the jupyter notebook that contains the basic logic for an AI that takes random actions:
```
http://127.0.0.1:8888/lab/tree/notebooks/00_random_agent.ipynb
```
You can run individual code snippets by pressing `SHIFT + ENTER`

#### 2.2. 👷‍♂️ With Local Setup

There are two ways to run the agent locally. Both methods achieve the same result - running an AI agent that takes random actions in the game:

#### **Method 1**
With this method, you need to have the game running with the AI interface enabled (using the `learning on` command). The agent will connect to the running game.

```
python scripts/00_random_agent.py
```

#### **Method 2**
This method launches the game automatically from your script. You don't need to have the game running beforehand, but you do need to provide the path to the game directory:

```
python scripts/00_random_agent.py --env_path game_directory --env_name DragonJump
```

You can find out your game_directory by opening the game, using the `~` (tilda) key to open the console, and typing `path`.

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

