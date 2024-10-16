# PLaiGROUND

PLaiGROUND (name may change) is a repository for learning Machine Learning by training Game AIs. 

## 🍽️ Table of Contents

Here are some nifty links to navigate this huge README file

- [Setup the project](#-setup-the-project) - What you need to do after you just cloned the repo
- [Running the project](#️-running-the-project) - How to start your first random agent
- [What's to come](#-whats-to-come) - A short ToDo for people to know that's coming
- [The GOATs](#-the-goats) - A shout out to all the people that helped along the way
- [Contact](#-contact) - In case you have any issues or just want to chat

## 👷 Setup the project
### 0. Install Python
This project uses **[Python 3.11](https://www.python.org/downloads/release/python-31110/)**. You can try an older version of Python, but you may have compatibility issues.

### 1. Create a virtual environment 

By default Python will install a project's dependencies (requirements) globally. This may cause compatibility issues if you have multiple Python projects on your PC. In order to avoid this, you will need to use a virtual environment that will store a project's dependencies separately.
```
# Install the virtualenv library globally
python -m pip install venv

# Create a virtual environment inside the current folder
python -m venv .venv
```

### 2. Activate the virtual environment

Depending on your operating system, you will need to run a specific command to step into the virtual environment.
```
# On Windows
./.venv/Scripts/activate

# On Linux and MacOS
source ./.venv/bin/activate
```
You'll know the command worked if the command line you'll see `(.venv)` at the beginning of the line.

### 3. Install dependencies

The project has several dependencies that were stored inside the `requirements.txt` file which are needed to run the code.

```
# Install the dependencies inside the virtual environment
pip install -r requirements.txt
```

## 🏃‍➡️ Running the project

### 1. Downloading the executable
You'll need to download the environment files that correspond to your Operating System [from here](https://github.com/2BytesGoat/ENV-DragonJump/releases). Those files could be placed inside the `./environments` after unpacking them.


### 2. Running the random agent script
To check that everything works as expected run the following command, after you **activated your virtual environment**.

```
# Start random agent
python 00_random_agent.py --env_path ./environments/ENV-DragonJump
```

The game will automatically stop after the Agent has done 1000 steps. In case you want to continue after the Agent is done, checkout the code and reset the environment.

### 3. Troubleshooting

**Corrupted application on MacOS** - the application was not signed during the build process because I didn't want to pay 99$ to apple. Because of that the OS will say that the application is corrupted. To fix this, you'll need to run the following commands:

```
# Navigate to the location of you .app file
cd ./environments/

# Add a certificate to the app
xattr -c ENV-DragonJump.app

# Go back to the root of the project
cd ../
```
After that the application should run smoothly.

## 📋 What's to come:

- [x] ~~find a place to store the environment files at~~
- [x] ~~add a link to download the env files from~~
- [ ] update godot library to support input configurations
- [ ] do a PR for godot library changes on official library git

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

