"""Greedy algorithm"""

import tensorflow as tf
import numpy as np
import pandas as pd

class Greedy:
    def __init__(self, env, args):
        self.env = env
        
        self.episode_max_steps = int(args["episode_length"] / args["time_step_size"])
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.max_waiting_time = int(args["max_waiting_time"] / args["time_step_size"])
        self.cost_parameter = args["cost_parameter"]
        self.data_dir = args["data_dir"]
        self.results_dir = args["results_dir"]
        
        self.graph = pd.read_csv(self.data_dir + f'/graph(dt_{args["time_step_size"]}s).csv', index_col=[0,1], usecols=[0,1,2,3,5])
    
        zone_IDs = self.graph.index.unique(level=0).tolist()
        idx = pd.MultiIndex.from_tuples([(i,i) for i in zone_IDs], names=["origin_ID", "destination_ID"])
        additional_entries = pd.DataFrame([[0,0] for i in range(len(zone_IDs))], idx, columns=["distance", "travel_time"])
        self.graph = self.graph.append(additional_entries)
        
        self.max_time = self.graph.travel_time.max()

    def __call__(self):
        test_rewards = []
        no_accepted_requests = 0
        
        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv')
        test_dates = test_dates.test_dates.tolist()
        count_test_dates = len(test_dates)
        
        for i in range(count_test_dates):
            test_reward = 0.
            obs, hvs = self.env.reset(testing=True)
            
            for j in range(self.episode_max_steps):
                if tf.reduce_all(obs["requests_state"] == tf.zeros([self.n_req_max, 5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                else:
                    action = self.get_action(obs, hvs)
                
                next_obs, reward, hvs = self.env.step(action)
                test_reward += tf.reduce_sum(reward).numpy()
                no_accepted_requests += tf.reduce_sum(tf.cast(reward != 0, tf.float16)).numpy()
                obs = next_obs
            
            test_rewards.append(test_reward)

        pd.DataFrame({"test_rewards_greedy": test_rewards}, index=test_dates).to_csv(self.results_dir + f"/test_rewards_greedy_{self.n_veh}veh.csv")
        
        with open(self.results_dir + f"/avg_test_reward_greedy_{self.n_veh}veh.txt", 'w') as f:
            f.write(str(np.mean(test_rewards)))
         
        with open(self.results_dir + f"/no_accepted_requests_heuristic_{self.n_veh}veh.txt", 'w') as f:
            f.write(str(no_accepted_requests))

    def get_action(self, state, hvs):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        not_dummy_request = tf.reduce_sum(requests_state, axis=1) > 0
        available_vehicles = (hvs[:,6] == -1).numpy()
        vehicle_locations = np.where(hvs[:,7] != -1,
                                     hvs[:,7], 
                                     np.where(hvs[:,4] != -1,
                                              hvs[:,4],
                                              hvs[:,0]))
        time_to_locations = vehicles_state[:,2].numpy() * self.max_time
        action = []
        for i in range(self.n_req_max):
            if not_dummy_request[i] & np.any(available_vehicles):
                pickup_ID = self.env.requests.iloc[i, 1]
                dropoff_ID = self.env.requests.iloc[i, 2]
                fare = self.graph.loc[(pickup_ID, dropoff_ID), "fare"]
                times_to_pickup = self.graph.loc[zip(vehicle_locations, [pickup_ID for i in range(self.n_veh)]), "travel_time"].to_numpy() + time_to_locations
                cond_wait_time = times_to_pickup <= self.max_waiting_time
                distance = self.graph.loc[zip(vehicle_locations, [pickup_ID for i in range(self.n_veh)]), "distance"].to_numpy() + self.graph.loc[(pickup_ID, dropoff_ID), "distance"]
                cond_profit = fare + distance * self.cost_parameter > 0
                eligible_vehicles = np.all([available_vehicles, cond_wait_time, cond_profit], axis=0)
                if np.any(eligible_vehicles):
                    distances = self.graph.loc[zip(vehicle_locations, [pickup_ID for i in range(self.n_veh)]), "distance"].to_numpy() # distance from pickup_ID for all vehicles
                    adapted_distances = np.where(eligible_vehicles, distances, 1000000) # set distance for not available vehicles to large number
                    a = np.argmin(adapted_distances)
                    available_vehicles[a] = False
                else:
                    a = -1
            else:
                a = -1
            action.append(a)
        return tf.constant(action)
