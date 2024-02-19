# Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems

This software uses a combination of multi-agent Soft Actor-Critic and weighted bipartite matching to train and test a policy, represented by a neural network, that dispatches vehicles to requests in an autonomous mobility on demand system. 

This method is proposed in:

> Tobias Enders, James Harrison, Marco Pavone, Maximilian Schiffer (2023). Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems. Proceedings of The 5th Annual Learning for Dynamics and Control Conference (L4DC 2023), in Proceedings of Machine Learning Research 211:1284-1296. Available from https://proceedings.mlr.press/v211/enders23a.html.

All components (code, data, etc.) required to run the experiments reported in the paper are provided here. This includes the greedy benchmark algorithm.

## Overview
The directory `algorithms` contains:
- The environment implementation in `environment.py`.
- The greedy benchmark algorithm in `greedy.py`, which can be executed using `main_greedy.py` with arguments as the exemplary ones in `args_greedy_XX_small/large_zones.txt` (see comments in `main_greedy.py` for explanations of the arguments).
- The remaining code files implement the hybrid multi-agent Soft Actor-Critic algorithm, which can be executed using `main.py` with arguments as the exemplary ones in `args_RL_XX_small/large_zones.txt` (see comments in `main.py` for explanations of the arguments). The code in `trainer.py` and `sac_discrete.py` is partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl).

The directory `data` contains pre-processed data for the two problem instances considered in the paper.


## Code Execution
To run the code with arguments `args.txt`, execute `python main.py @args.txt` (analogously for the greedy algorithm). 

For typical instance and neural network sizes, a GPU should be used. 
