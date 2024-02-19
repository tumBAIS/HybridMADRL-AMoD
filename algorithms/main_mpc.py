"""Model predictive control (MPC) algorithm"""

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from environment import Environment
from mip import ResourceConstrainedFlowModel


# parse parameters
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--num_seeds', type=int)  # for how many seeds runs with MPC algorithm are repeated
parser.add_argument('--episode_length', type=int)  # episode length in seconds
parser.add_argument('--time_step_size', type=int)  # time step size in seconds
parser.add_argument('--smoothed_prob_dist', type=str)  # whether Laplace smoothed probability distribution is used (from which future requests are sampled)
parser.add_argument('--dist_interval', type=int)  # length of time interval for which one distribution is computed, in seconds
parser.add_argument('--sampling_horizon', type=int)  # horizon for which future requests are sampled, in seconds
parser.add_argument('--veh_count', type=str)  # no. of vehicles (may be a comma-separated list to run multiple instances at once)
parser.add_argument('--max_req_count', type=int)  # max. no. of requests per time step
parser.add_argument('--max_waiting_time', type=int)  # max. waiting time in seconds
parser.add_argument('--cost_parameter', type=float)  # mileage-dependent cost for maintenance etc. in USD per meter
parser.add_argument('--local_rew_share', type=float)  # share of reward that is allocated locally (vs. global share, i.e., egoistic vs. system rewards)
parser.add_argument('--data_dir', type=str)  # relative path to directory where data is stored
parser.add_argument('--results_dir', type=str)  # relative path to directory where results shall be saved

args = vars(parser.parse_args())

if args["smoothed_prob_dist"] == "False":
    args["smoothed_prob_dist"] = False
elif args["smoothed_prob_dist"] == "True":
    args["smoothed_prob_dist"] = True
else:
    raise argparse.ArgumentTypeError('True or False expected for argument --smoothed_prob_dist.')

num_seeds = args["num_seeds"]
dt = args["time_step_size"]
smoothed_prob_dist = args["smoothed_prob_dist"]
time_steps_count = int(args["episode_length"]/dt)
n_req_max = args["max_req_count"]
vehicles_list = [int(i) for i in args["veh_count"].split(',')]
max_waiting_time = int(args["max_waiting_time"] / dt)
cost_parameter = args["cost_parameter"]
data_dir = args["data_dir"]
results_dir = args["results_dir"]
dist_interval = args["dist_interval"]
sampling_horizon = args["sampling_horizon"]

# get list of test dates
test_dates = pd.read_csv(data_dir + '/test_dates.csv')
test_dates = test_dates.test_dates.tolist()
count_test_dates = len(test_dates)

# get graph
graph = pd.read_csv(data_dir + f'/graph(dt_{dt}s).csv', header=0, index_col=[0,1], usecols=[0,1,2,3,5])
zone_IDs = graph.index.unique(level=0).tolist()
idx = pd.MultiIndex.from_tuples([(i,i) for i in zone_IDs], names=["origin_ID", "destination_ID"])
additional_entries = pd.DataFrame([[0,0,0] for i in range(len(zone_IDs))], idx, columns=["distance", "travel_time", "fare"])
graph = graph.append(additional_entries)
max_time = graph.travel_time.max()

# get previously estimated request probability distribution per origin-destination-pair and average request count
if smoothed_prob_dist:
    od_prob_dist = np.load(data_dir + '/od_prob_dist_smoothed.npy')
else:
    od_prob_dist = np.load(data_dir + '/od_prob_dist.npy')
avg_req_count = np.load(data_dir + '/avg_req_count.npy')
req_count_factor = sampling_horizon / dist_interval
avg_req_count = [round(i*req_count_factor) for i in avg_req_count]

for num_vehicles in vehicles_list:
    args["veh_count"] = num_vehicles

    def get_action(state, hvs, real_requests, step):
        veh_pos = np.where(hvs[:,7] != -1,
                           hvs[:,7], 
                           np.where(hvs[:,4] != -1,
                                    hvs[:,4],
                                    hvs[:,0]))
        
        time_to_pos = state["vehicles_state"][:,2].numpy() * max_time
        
        ### requests
        # real requests
        real_requests = real_requests[["pickup_time","pickup_ID","dropoff_ID","distance"]]
        real_requests["pickup_time"] = 0
        
        # sampling of future requests
        cur_interval = int(step/(dist_interval/dt))
        count_sampled_req = avg_req_count[cur_interval]
        
        dist = od_prob_dist[cur_interval,:,:]
        flat_prstate = dist.flatten()
        sample = np.random.choice(a=flat_prstate.size, p=flat_prstate, size=count_sampled_req)
        sampled_o, sampled_d = np.unravel_index(sample, dist.shape)
        
        sampled_pickup_time = np.sort(np.random.choice(a=np.arange(1,int(sampling_horizon/dt)+1),size=count_sampled_req))
        
        sampled_requests = pd.DataFrame({"pickup_time": sampled_pickup_time,
                                         "pickup_ID": sampled_o,
                                         "dropoff_ID": sampled_d})
        sampled_requests["distance"] = pd.MultiIndex.from_arrays([sampled_requests.pickup_ID, sampled_requests.dropoff_ID])
        sampled_requests.distance = sampled_requests.distance.map(graph.distance)
    
        # concat and add additional columns
        requests = pd.concat([real_requests,sampled_requests], ignore_index=True)
            
        requests["dropoff_time"] = pd.MultiIndex.from_arrays([requests.pickup_ID, requests.dropoff_ID])
        requests.dropoff_time = requests.dropoff_time.map(graph.travel_time)
        requests["dropoff_time"] = requests.pickup_time + requests.dropoff_time
        
        requests["trip_profit"] = pd.MultiIndex.from_arrays([requests.pickup_ID, requests.dropoff_ID])
        requests.trip_profit = requests.trip_profit.map(graph.fare)
        requests.trip_profit = requests.trip_profit + cost_parameter * requests.distance
        
        ### derived parameters
        num_requests = len(requests.index)
        num_vertices = 2 + num_vehicles + num_requests
        sink_index = num_vertices - 1
        
        ### time windows and service times for nodes
        # vertex indices:
        # - source: 0
        # - vehicles: 1, ..., num_vehicles
        # - requests: num_vehicles + 1, ..., num_vehicles + num_requests
        # - sink: num_vehicles + num_requests + 1
        tws = []
        service_times = []
        
        # source
        tws.append((0,100))
        service_times.append(0)
        
        # vehicles
        for i in range(num_vehicles):
            tws.append((0,100))
            service_times.append(time_to_pos[i])
        
        # requests
        for i in range(num_requests): 
            start = int(requests.iloc[i,0])
            end = start + max_waiting_time
            service_time = graph.loc[(requests.iloc[i,1], requests.iloc[i,2]), "travel_time"]
            
            tws.append((start,end))
            service_times.append(service_time)
        
        # sink  
        tws.append((0,100))
        service_times.append(0)
            
        ### edges
        arcs = []
        
        # source -> vehicles
        for vehicle in range(1, num_vehicles + 1):
            arcs.append((0,vehicle,0,0))
        
        # vehicles -> sink
        for vehicle in range(1, num_vehicles + 1):
            arcs.append((vehicle,sink_index,0,0))
        
        # vehicles -> requests
        vehs = pd.DataFrame({"pos": veh_pos,
                             "time_to_pos": time_to_pos,
                             "veh_ID": range(num_vehicles)})
        
        reqs = requests.copy()
        reqs = reqs[["pickup_time", "pickup_ID", "trip_profit"]]
        reqs["req_ID"] = range(num_requests)
        
        cross_veh_req = vehs.merge(reqs, how='cross')
        
        cross_veh_req["time_pos_o"] = graph.loc[zip(cross_veh_req.pos, cross_veh_req.pickup_ID), "travel_time"].to_numpy()
        
        cond1 = (cross_veh_req.time_pos_o <= max_waiting_time).to_numpy()
        cond2 = (cross_veh_req.time_to_pos + cross_veh_req.time_pos_o <= cross_veh_req.pickup_time + max_waiting_time).to_numpy()
        cross_veh_req = cross_veh_req[cond1 & cond2]
        
        cross_veh_req["start"] = 1 + cross_veh_req.veh_ID
        cross_veh_req["end"] = num_vehicles + 1 + cross_veh_req.req_ID
        cross_veh_req["weight"] = cross_veh_req.trip_profit + cost_parameter * graph.loc[zip(cross_veh_req.pos, cross_veh_req.pickup_ID), "distance"].to_numpy()
        cross_veh_req.weight = (cross_veh_req.weight * (-100)).round().astype(int)
        cross_veh_req = cross_veh_req[["start", "end", "weight", "time_pos_o"]]
        
        arcs = arcs + list(cross_veh_req.to_records(index=False))
    
        # requests -> requests
        request1 = requests.copy()
        request1 = request1[["pickup_time","pickup_ID","dropoff_ID","dropoff_time","trip_profit"]]
        request1["req_ID"] = range(num_requests)
        request2 = request1.copy()
        request1.rename(columns={"pickup_time": "pickup_time1",
                                 "pickup_ID": "pickup_ID1",
                                 "dropoff_ID": "dropoff_ID1",
                                 "dropoff_time": "dropoff_time1",
                                 "trip_profit": "trip_profit1",
                                 "req_ID": "req_ID1"}, 
                        inplace=True)
        request2.rename(columns={"pickup_time": "pickup_time2",
                                 "pickup_ID": "pickup_ID2",
                                 "dropoff_ID": "dropoff_ID2",
                                 "dropoff_time": "dropoff_time2",
                                 "trip_profit": "trip_profit2",
                                 "req_ID": "req_ID2"}, 
                        inplace=True)
        crossed_requests = request1.merge(request2, how='cross')

        cond1 = (crossed_requests.pickup_time1 < crossed_requests.pickup_time2).to_numpy()
        cond2 = (graph.loc[zip(crossed_requests.dropoff_ID1, crossed_requests.pickup_ID2), "travel_time"] <= max_waiting_time).to_numpy()
        cond3 = (crossed_requests.dropoff_time1.to_numpy() + graph.loc[zip(crossed_requests.dropoff_ID1, crossed_requests.pickup_ID2), "travel_time"].to_numpy() <= crossed_requests.pickup_time2.to_numpy() + max_waiting_time)
        crossed_requests = crossed_requests[cond1 & cond2 & cond3]
        
        crossed_requests["start"] = num_vehicles + 1 + crossed_requests.req_ID1
        crossed_requests["end"] = num_vehicles + 1 + crossed_requests.req_ID2
        crossed_requests["weight"] = crossed_requests.trip_profit2 + cost_parameter * graph.loc[zip(crossed_requests.dropoff_ID1, crossed_requests.pickup_ID2), "distance"].to_numpy()
        crossed_requests.weight = (crossed_requests.weight * (-100)).round().astype(int)
        crossed_requests["travel_time"] = graph.loc[zip(crossed_requests.dropoff_ID1, crossed_requests.pickup_ID2), "travel_time"].to_numpy()
        crossed_requests = crossed_requests[["start", "end", "weight", "travel_time"]]
        
        arcs = arcs + list(crossed_requests.to_records(index=False))
        
        # requests -> sink
        for request in range(num_vehicles + 1, num_vehicles + 1 + num_requests):
            arcs.append((request,sink_index,0,0))
        
        ### construct and solve the model
        model = ResourceConstrainedFlowModel(arcs=arcs, tws=tws, service_times=service_times)
        solution, bound, selected_arcs = model.optimize()

        ### get action from model solution
        selected_arcs_starts = np.array([u for (u,v) in selected_arcs])
        selected_arcs_ends = np.array([v for (u,v) in selected_arcs], dtype=int)
        real_req_indices = np.arange(num_vehicles + 1, num_vehicles + len(real_requests.index) + 1)
        selected_arcs_to_real_requests_filter = np.isin(selected_arcs_ends, real_req_indices)
        selected_arcs_to_real_requests_starts = selected_arcs_starts[selected_arcs_to_real_requests_filter]
        selected_arcs_to_real_requests_ends = selected_arcs_ends[selected_arcs_to_real_requests_filter]
        selected_arcs_to_real_requests_starts -= 1
        selected_arcs_to_real_requests_ends -= (num_vehicles + 1)
        
        act = -np.ones(n_req_max, int)
        act[selected_arcs_to_real_requests_ends] = selected_arcs_to_real_requests_starts
        
        act = tf.constant(act)
        
        if solution != 0:
            gap = bound / solution - 1
        else:
            gap = 0.
        
        return act, gap
    
    # run model predictive control for all test dates num_seeds many times for the respective number of vehicles
    # and save rewards (per test date and on average)
    rews = np.zeros((num_seeds, count_test_dates))
    total_gaps = np.zeros((num_seeds, count_test_dates))
    
    for seed in np.arange(0, num_seeds):
        env = Environment(args)

        for i in range(count_test_dates):
            rew = 0.
            total_gap = 0.
            gap_counter = 0
            
            state, hvs = env.reset(testing=True)
            
            for j in range(time_steps_count):
                print("\n\n\n ========================================================\n",
                      num_vehicles, " vehicles, seed ", seed, ", episode", i+1, ", step", j+1, "\n\n")
                if tf.reduce_all(state["requests_state"] == tf.zeros([n_req_max,5])):
                    action = -tf.ones(n_req_max, tf.int32)
                
                else:
                    action, gap = get_action(state, hvs, env.requests, j)
                    total_gap += gap
                    gap_counter += 1
                
                next_state, reward, hvs = env.step(action)
                
                rew += tf.reduce_sum(reward).numpy()
                
                state = next_state
                
            total_gap /= gap_counter
            
            rews[seed, i] = rew
            total_gaps[seed, i] = total_gap
        
        pd.DataFrame({"test_rewards_MPC": rews[seed, :]}, index=test_dates).to_csv(results_dir + f"/test_rewards_MPC_{num_vehicles}veh_seed{seed}.csv")
        pd.DataFrame({"gaps": total_gaps[seed, :]}, index=test_dates).to_csv(results_dir + f"/gaps_{num_vehicles}veh_seed{seed}.csv")
    
    mean_rew_per_date = np.mean(rews, axis=0)
    mean_rew_per_seed = np.mean(rews, axis=1)
    mean_rew = np.mean(rews)
    
    mean_gap_per_date = np.mean(total_gaps, axis=0)
    mean_gap_per_seed = np.mean(total_gaps, axis=1)
    mean_gap = np.mean(total_gaps)
    
    pd.DataFrame({"test_rewards_MPC": mean_rew_per_date}, index=test_dates).to_csv(results_dir + f"/mean_rew_per_date_{num_vehicles}veh.csv")
    pd.DataFrame({"gaps": mean_gap_per_date}, index=test_dates).to_csv(results_dir + f"/mean_gap_per_date_{num_vehicles}veh.csv")
    
    pd.DataFrame({"test_rewards_MPC": mean_rew_per_seed}, index=np.arange(0, num_seeds)).to_csv(results_dir + f"/mean_rew_per_seed_{num_vehicles}veh.csv")
    pd.DataFrame({"gaps": mean_gap_per_seed}, index=np.arange(0, num_seeds)).to_csv(results_dir + f"/mean_gap_per_seed_{num_vehicles}veh.csv")
    
    with open(results_dir + f"/avg_test_reward_MPC_{num_vehicles}veh.txt", 'w') as f:
        f.write(str(mean_rew))
    with open(results_dir + f"/avg_gap_{num_vehicles}veh.txt", 'w') as f:
        f.write(str(mean_gap))
    