# Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems

This software uses a combination of multi-agent Soft Actor-Critic and weighted bipartite matching to train and test a policy, represented by a neural network, that dispatches vehicles to requests in an autonomous mobility on demand system. 

This method is proposed in:

> Tobias Enders, James Harrison, Marco Pavone, Maximilian Schiffer (2023). Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems. Proceedings of The 5th Annual Learning for Dynamics and Control Conference (L4DC 2023), in Proceedings of Machine Learning Research 211:1284-1296. Available from https://proceedings.mlr.press/v211/enders23a.html.

All components (code, data, etc.) required to run the code for the instances considered in the paper are provided here. 

## Overview
The directory `algorithms` contains:
- The code (`trainer.py` and `sac_discrete.py` are partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl))
- An argument file `args_RL_XX_small/large_zones.txt` for each problem instance (see comments in `main.py` for explanations of the arguments)

The directory `data` contains pre-processed data for the two problem instances considered in the paper.


## Code Execution
To run the code with arguments `args.txt`, execute `python main.py @args.txt`. 

For typical instance and neural network sizes, a GPU should be used. 
