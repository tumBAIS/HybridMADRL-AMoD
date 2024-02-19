"""Parse parameters and run greedy algorithm"""

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--episode_length', type=int) # episode length in seconds
parser.add_argument('--time_step_size', type=int) # time step size in seconds
parser.add_argument('--veh_count', type=str) # no. of vehicles (may be a comma-separated list to run multiple instances at once)
parser.add_argument('--downscaling_factor', type=int) # downscaling factor for trip data (not needed for algorithm, only included to keep track of it)
parser.add_argument('--max_req_count', type=int) # max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int) # max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float) # mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--local_rew_share', type=float) # share of reward that is allocated locally (vs. global share, i.e., egoistic vs. system rewards)
parser.add_argument('--data_dir', type=str) # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str) # relative path to directory where results shall be saved

args = vars(parser.parse_args())

vehicles_list = [int(i) for i in args["veh_count"].split(',')]

from environment import Environment
from greedy import Greedy

for veh_count in vehicles_list:
    print(veh_count, "vehicles")
    args["veh_count"] = veh_count
    
    env = Environment(args)
    greedy = Greedy(env, args)

    greedy()
