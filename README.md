# Drone Routing Problems

## Table of Contents

* [About the Project](#about-the-project)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Environment](#environment)
* [File Structure](#file-structure)
* [Using Epymarl](#using-epymarl)
* [Training Parameters](#training-parameters)

## About the Project

A multi-agent reinforcement learning environment for drone routing problems (DRP), compatible with [epymarl](https://github.com/uoe-agents/epymarl).

## Prerequisites

Before starting, you need the following tools installed on your machine.

### Terminal (command line)

* **Mac**: Open `Terminal` from Applications > Utilities. bash/zsh is available by default — no installation needed.
* **Windows**: Use `Command Prompt` or `PowerShell`. For a better experience, consider installing [Git for Windows](https://gitforwindows.org/) which includes Git Bash.

### Anaconda (Python environment manager)

Anaconda is required to manage Python versions and packages.

1. Download the installer from: [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Run the installer and follow the on-screen instructions
3. After installation, open a new terminal and verify it works:

```bash
conda --version
```

If you see a version number (e.g. `conda 23.x.x`), the installation was successful.

### Git

Git is required to download this repository.

* **Mac**: Run `git --version` in the terminal. If not installed, macOS will prompt you to install it automatically.
* **Windows**: Download from [https://git-scm.com/](https://git-scm.com/)

## Installation

### 1. Download this repository

Open a terminal and run:

```bash
git clone https://github.com/Yamaguchi-yushi/MARL4DRP.git
cd MARL4DRP
```

### 2. Create and activate a conda environment

```bash
conda create -n env_name python=3.9
conda activate env_name
```

> Replace `env_name` with any name you like (e.g. `drp_env`).
> After activation, you should see `(env_name)` at the beginning of your terminal prompt.

### 3. Install all dependencies

Make sure you are in the root directory of this repository (the folder containing `install.sh`), then run:

```bash
bash install.sh
```

> **Note:** Warnings about `conda-libmamba-solver` can be safely ignored.
> `bash` is available by default on Mac and Linux. On Windows, use Git Bash or WSL.

## Environment

How to create environments with the gym framework:

```python
import gym
import drp_env
env = gym.make("drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

or

```python
import gym
env = gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag="onehot_fov")
```

### Environment name

```text
drp-{agent_num}agent_{map_name}-v2
```

* agent_num: number of agents, 1~6
* map_name: `map_3x3` / `map_5x4` / `map_8x5` / `map_10x6` / `map_10x8` / `map_10x10` / `map_aoba00` / `map_aoba01`
* state_repre_flag: kind of observation — `coordinate` / `onehot` / `onehot_fov` / `heu_onehot` / `heu_onehot_fov`

### Action

Node number. When taking an invalid action, the agent stops at its current position.

### Observation

...

### Reward

Rewards are set per agent, determined by `reward_list`.

Default:

```yaml
reward_list:
  goal: 100
  collision: -10
  wait: -10
  move: -1
```

* **goal**: When an agent reaches its goal, `reward = reward_list["goal"]`
* **collision**: When agents collide, `reward = reward_list["collision"] * speed (default: 5)`
* **wait**: When an agent stops at a non-goal position, `reward = reward_list["wait"] * speed (default: 5)`
* **move**: When an agent moves, `reward = reward_list["move"] * speed (default: 5)`

### Info

* **distance_from_start** (List of float): Distance traveled from start node for each agent. Increases by `speed` when an agent stops or collides.
* **goal** (Bool): `True` when all agents have reached their goals.
* **collision** (Bool): `True` when agents collide.
* **timeup** (Bool): `True` when the time limit is reached (used by epymarl).
* **cost** (int): Episode cost. Lower is better. Equals the sum of arrival steps on success, or `agent_num * time_limit` on failure.
* **goal_cost** (int or None): Same as `cost` when all agents reach their goals; `None` otherwise.

## File Structure

```text
MARL4DRP
├── README.md
├── install.sh
├── requirements.txt
├── setup.py
├── drp_env
│   ├── __init__.py
│   ├── drp_env.py
│   ├── EE_map.py
│   ├── map
│   └── state_repre
├── drpload_test.py
├── for_epymarl
└── epymarl
```

Name                              |  Description
---------------------------------- | ------------------------------------------------------------------------------------
drp\_env                          |  the directory for package drp\_env
drpload\_test.py                  |  a sample file using drp\_env
for\_epymarl                      |  files required to work with epymarl
epymarl                           |  multi-agent RL framework (epymarl v1.0.0)
install.sh                        |  installation script for all dependencies

Directories/files in drp\_env:

Name                              |  Description
---------------------------------- | ------------------------------------------------------------------------------------
\_\_init\_\_.py                   |  register environments
drp\_env.py                       |  environment with gym structure
EE\_map.py                        |  process related to network structure
map                               |  csv files about map information
state\_repre                      |  manage observation of environments

## Using Epymarl

The `epymarl` folder in this repository already contains the modified version of epymarl configured to work with `drp_env`.

Run training from the `epymarl` directory:

```bash
cd epymarl
python3 src/main.py --config=iql --env-config=gymma with env_args.time_limit=100 'env_args.key=drp_env:drp-1agent_map_3x3-v2' env_args.state_repre_flag="onehot"
```

> **Note:** Use single quotes `'` around arguments containing `:` to avoid shell parsing issues.

### Available algorithms

Replace `--config=iql` with any of the following:

Config  | Algorithm
-------- | ----------
`iql`   | Independent Q-Learning
`qmix`  | QMIX
`vdn`   | Value Decomposition Networks


## Training Parameters

All experiments were conducted using QMIX with the following hyperparameters.

### Algorithm (QMIX)

Parameter | Value
--- | ---
Optimizer | Adam
Learning rate | 0.0005
Discount factor (γ) | 0.99
Batch size | 32 episodes
Replay buffer size | 5,000
Hidden dim | 64
Mixing embed dim | 32
Hypernet layers | 2
Hypernet embed | 64
Target update interval | 200 steps
Double Q-learning | True
Gradient norm clip | 10

### Exploration

Parameter | Value
--- | ---
Strategy | ε-greedy
ε start | 1.0
ε finish | 0.05
ε anneal time | 50,000 steps

### LaRe (LLM-based Reward Decomposition)

Parameter | Value | Description
--- | --- | ---
`factor_reward_model_layers` | 3 | Number of layers in factor reward model
Replay buffer size (LaRe) | 1,024 | Single shared memory
Reward model update freq | 256 | Frequency of reward model updates
Reward model batch size | 256 | Batch size for reward model training
Reward model warmup | 256 samples | Minimum samples before training starts
Reward model lr | 5e-4 | Learning rate for reward model
Evaluation episodes | 16 | Episodes used to evaluate reward model

### Experimental Conditions

Map | Agents | Total steps | Time limit
--- | --- | --- | ---
map_8x5 | 3 | 4.0M | 100
map_8x5 | 4 | 8.0M | 100
map_8x5 | 5 | 15.0M | 100
map_aoba00 | 3 | 15.0M | 200
map_aoba00 | 4 | 25.0M | 200
map_aoba00 | 5 | 30.0M | 200
