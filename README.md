# Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems

This software uses a combination of multi-agent Soft Actor-Critic and weighted bipartite matching to train and test a policy, represented by a neural network, that dispatches vehicles to requests in an autonomous mobility on demand system. 

This method is proposed in:

> Tobias Enders, James Harrison, Marco Pavone, and Maximilian Schiffer. Hybrid Multi-agent Deep Reinforcement Learning for Autonomous Mobility on Demand Systems. arXiv preprint [arXiv:2212.07313](https://arxiv.org/abs/2212.07313), 2022.

All components (code, data, etc.) required to run the code for the instances considered in the paper are provided here. 

## Overview
The directory `code and data` contains:
- The code (`trainer.py` and `sac_discrete.py` are partly based on code from this [GitHub repository](https://github.com/keiohta/tf2rl))
- Pre-processed data for the two problem instances considered in the paper
- An argument file `args.txt` for each problem instance (see comments in `main.py` for explanations of the arguments)

## Code Execution
To run the code for an instance with data and arguments `args.txt` saved in `data_dir`, execute `python main.py @data_dir/args.txt`. 

For typical instance and neural network sizes, a GPU should be used. 
