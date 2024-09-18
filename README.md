# VF-ICE
 VF based on Individual contribution Evaluation Mechanism

# Research Paper and environment

QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

The StarCraft Multi-Agent Challenge : Environment Code

The StarCraft Multi-Agent Challenge : Research Paper

# setup

Using Python 3.6.

Pytorch 1.7.1

tensorflow 2.0.

tensorboard 2.0.

Anaconda 3.

ubutun 20.4.

Be sure to set up the environment variable : SC2PATH (see lauch.bat)

# Train an AI

python main.py --scenario [scenario_name] --train

# Test an AI

python main.py --scenario [scenario_name]


# Please note:

At runtime, place logs and model in the VF-ICE folder

There are five algorithms, including IQL, VDN, QMIX, PCQMIX and VF-ICE

Please select the algorithm name in main. py
