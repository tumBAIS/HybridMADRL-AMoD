"""Parse parameters and run algorithm"""

# parse parameters
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--random_seed', type=int) # random seed
parser.add_argument('--episode_length', type=int) # episode length in seconds
parser.add_argument('--time_step_size', type=int) # time step size in seconds
parser.add_argument('--veh_count', type=int) # no. of vehicles
parser.add_argument('--downscaling_factor', type=int) # downscaling factor for trip data (not needed for algorithm, only included to keep track of it)
parser.add_argument('--max_req_count', type=int) # max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int) # max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float) # mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--max_steps', type=int) # no. of steps to interact with environment
parser.add_argument('--min_steps', type=int) # no. of steps before neural net weight updates begin
parser.add_argument('--random_steps', type=int) # no. of steps with random policy at the beginning
parser.add_argument('--update_interval', type=int) # no. of steps between neural net weight updates
parser.add_argument('--validation_interval', type=int) # no. of steps between validation runs (must be multiple of no. of time steps per episode)
parser.add_argument('--tracking_interval', type=int) # interval at which training data is saved 
parser.add_argument('--profile_interval', type=int) # interval at which training step is profiled
parser.add_argument('--post_processing', type=str, choices=['simple_matching', 'weighted_matching']) # whether final decisions are obtained through maximum bipartite matching or weighted bipartite matching
parser.add_argument('--attention', type=str) # whether attention layer is used
parser.add_argument('--req_embedding_dim', type=int) # units of request embedding layer
parser.add_argument('--veh_embedding_dim', type=int) # units of vehicle embedding layer
parser.add_argument('--req_context_dim', type=int) # first dim of W in request context layer
parser.add_argument('--veh_context_dim', type=int) # first dim of W in vehicle context layer
parser.add_argument('--inner_units', type=str) # units of inner network (sequence of feedforward layers)
parser.add_argument('--regularization_coefficient', type=float) # coefficient for L2 regularization of networks (0 if no regularization)
parser.add_argument('--rb_size', type=int) # replay buffer size
parser.add_argument('--batch_size', type=int) # (mini-)batch size
parser.add_argument('--log_alpha', type=float) # log(alpha)
parser.add_argument('--tau', type=float) # smoothing factor for exponential moving average to update target critic parameters
parser.add_argument('--huber_delta', type=float) # delta value at which Huber loss becomes linear
parser.add_argument('--gradient_clipping', type=str) # whether gradient clipping is applied
parser.add_argument('--clip_norm', type=float) # global norm used for gradient clipping
parser.add_argument('--scheduled_lr', type=str) # whether the learning rate follows a schedule
parser.add_argument('--lr', type=float) # learning rate (if scheduled, this is the start value)
parser.add_argument('--lr_end', type=float) # final learning rate if lr is scheduled
parser.add_argument('--lr_decay_steps', type=int) # step at which learning rate is set to final value if scheduled (referring to experience collection steps, not no. of weight updates)
parser.add_argument('--scheduled_discount', type=str) # whether discount factor follows a schedule
parser.add_argument('--discount', type=float) # discount factor (if scheduled, this is the start value)
parser.add_argument('--scheduled_discount_values', type=str) # list of discount factor values to be set at chosen steps (if scheduled)
parser.add_argument('--scheduled_discount_steps', type=str) # list of steps at which discount factor is set to chosen value (if scheduled)
parser.add_argument('--normalized_rews', type=str) # whether rewards are normalized when sampled from replay buffer (if so, they are divided by the standard deviation of rewards currently stored in the replay buffer)
parser.add_argument('--local_rew_share', type=float) # share of reward that is allocated locally (vs. global share, i.e., egoistic vs. system rewards)
parser.add_argument('--data_dir', type=str) # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str) # relative path to directory where results shall be saved
parser.add_argument('--model_dir', type=str, default=None) # relative path to directory with saved model that shall be restored in the beginning (overwriting default initialization of network weights)

args = vars(parser.parse_args())

args["inner_units"] = [int(i) for i in args["inner_units"].split(',')]
args["scheduled_discount_values"] = [float(i) for i in args["scheduled_discount_values"].split(',')]
args["scheduled_discount_steps"] = [int(i) for i in args["scheduled_discount_steps"].split(',')]

if args["attention"] == "False":
    args["attention"] = False
elif args["attention"] == "True":
    args["attention"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --attention.')

if args["gradient_clipping"] == "False":
    args["gradient_clipping"] = False
elif args["gradient_clipping"] == "True":
    args["gradient_clipping"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --gradient_clipping.')

if args["scheduled_lr"] == "False":
    args["scheduled_lr"] = False
elif args["scheduled_lr"] == "True":
    args["scheduled_lr"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --scheduled_lr.')

if args["scheduled_discount"] == "False":
    args["scheduled_discount"] = False
elif args["scheduled_discount"] == "True":
    args["scheduled_discount"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --scheduled_discount.')

if args["normalized_rews"] == "False":
    args["normalized_rews"] = False
elif args["normalized_rews"] == "True":
    args["normalized_rews"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --normalized_rews.')

# set seed and further global settings
seed = args["random_seed"]

import os
os.environ['PYTHONHASHSEED'] = str(seed)

os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2" # enable XLA

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

tf.keras.mixed_precision.set_global_policy('mixed_float16') # enable mixed precision computations

# initialize environment, Soft Actor-Critic and trainer classes
from environment import Environment
from sac_discrete import SACDiscrete
from trainer import Trainer

env = Environment(args)
policy = SACDiscrete(args, env)
trainer = Trainer(policy, env, args)

# call trainer
trainer()
