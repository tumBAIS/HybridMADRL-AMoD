""" Episode initialization, transition and reward function to get next states and rewards

Gives state encodings as needed for the neural networks and uses hvs (hidden vehicles state) to track vehicles' state internally

Definition of states, actions, rewards:

Hidden vehicles state: array of shape (vehicles count, 8)
-------------------------------------------------------------------------------
Entries per vehicle:
    v: int in (0,...,nodes count-1)
    tau: int >= 0
    omega(r1): int >= 0 or -1 (-1 if no request assigned or already picked up)
    o(r1): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    d(r1): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    omega(r2): int >= 0 or -1 (-1 if no second request assigned)
    o(r2): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)
    d(r2): int in (0,...,nodes count-1) or -1 (-1 if no request assigned)

State: Dictionary with entries "requests_state", "vehicles_state", "misc_state"
-------------------------------------------------------------------------------
Requests: padded tensor of shape (requests count max, 5)
    Origin:
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Destination:
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Distance between origin and destination, divided by max. distance between any two nodes in graph (scalar in [0,1])

Vehicles: tensor of shape (vehicles count, 4)
    Position: current node if no request assigned, otherwise destination of assigned request that will be served last
        horizontal idx, normalized to [0,1]
        vertical idx, normalized to [0,1]
    Time steps to reach position, divided by max. time between any two nodes in graph (scalar >= 0, usually < 1)
    Number of assigned requests, divided by 2 (scalar in [0,1])

Misc: tensor of shape (3,)
    Time step, divided by no. of time steps in one episode (scalar in [0,1])
    Count of requests placed since start of current episode, divided by count of requests placed on average until current time step (scalar >= 0, usually close to 1)
    Sum over all vehicles of time steps to reach position, divided by number of vehicles x max. time between any two nodes in graph x 4 (scalar in [0,1])

Action: padded tensor of shape (requests count max,)
-------------------------------------------------------------------------------
Vehicle index to which request is assigned, -1 if rejected

Reward: padded tensor of shape (vehicles count * requests count max,)
-------------------------------------------------------------------------------
Each entry is a float representing the reward for the respective agent (agents in the order vehicle 1/request 1, ..., vehicle n_veh/request 1, vehicle1/request2, ...)
"""

import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf


class Environment(object):
    def __init__(self, args):
        self.episode_length = args["episode_length"]
        self.dt = args["time_step_size"]
        self.time_steps_count = tf.constant(int(self.episode_length/self.dt))
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.max_waiting_time = int(args["max_waiting_time"] / self.dt) # convert from seconds to time steps
        self.cost_parameter = args["cost_parameter"]
        self.local_rew_share = args["local_rew_share"]
        self.global_rew_share = 1 - self.local_rew_share
        self.data_dir = args["data_dir"]
        
        training_dates = pd.read_csv(self.data_dir + '/training_dates.csv')
        validation_dates = pd.read_csv(self.data_dir + '/validation_dates.csv')
        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv')
        self.training_dates = training_dates.training_dates.tolist()
        self.validation_dates = validation_dates.validation_dates.tolist()
        self.test_dates = test_dates.test_dates.tolist()
        self.remaining_training_dates = copy.deepcopy(self.training_dates)
        self.remaining_validation_dates = copy.deepcopy(self.validation_dates)
        
        zones = pd.read_csv(self.data_dir + '/zones.csv', header=0, index_col=0)
        self.max_horizontal_idx = zones.horizontal_idx.max()
        self.max_vertical_idx = zones.vertical_idx.max()
        self.horizontal_idx_table = self.get_lookup_table(tf.constant(zones.index, dtype=tf.int32), tf.constant(zones.horizontal_idx, dtype=tf.int32))
        self.vertical_idx_table = self.get_lookup_table(tf.constant(zones.index, dtype=tf.int32), tf.constant(zones.vertical_idx, dtype=tf.int32))
        keys = self.get_keys(zones.horizontal_idx, zones.vertical_idx)
        self.zone_mapping_table = self.get_lookup_table(keys, tf.constant(zones.index, dtype=tf.int32)) # lookup table for mapping from horizontal/vertical idx to zone ID
        
        self.graph = pd.read_csv(self.data_dir + f'/graph(dt_{self.dt}s).csv', index_col=[0,1])
        self.graph.route = self.graph.route.apply(lambda x: list(map(int, x[1:-1].split(', '))))
        self.graph["next_node"] = np.array([self.graph.route.tolist()[i][1] for i in range(len(self.graph.index))])
        
        self.nodes_count = len(self.graph.index.unique(level=0))
        self.max_distance = self.graph.distance.max()
        self.max_time = self.graph.travel_time.max()
        
        distances = tf.constant(self.graph["distance"].reset_index(), tf.int32)
        travel_times = tf.constant(self.graph["travel_time"].reset_index(), tf.int32)
        fares = tf.constant(self.graph["fare"].reset_index(), tf.float32)
        next_nodes = tf.constant(self.graph["next_node"].reset_index(), tf.int32)
        keys = self.get_keys(distances[:,0], distances[:,1])
        self.distance_table = self.get_lookup_table(keys, distances[:,2])
        self.travel_time_table = self.get_lookup_table(keys, travel_times[:,2])
        self.fare_table = self.get_lookup_table(keys, fares[:,2])
        self.next_node_table = self.get_lookup_table(keys, next_nodes[:,2])
        
        self.avg_request_count = pd.read_csv(self.data_dir + f'/avg_request_count_per_timestep(dt_{self.dt}s).csv', header=None, names=["avg_count"])
        self.avg_request_count = tf.constant(self.avg_request_count.avg_count.tolist())

    def get_keys(self, x, y):
        return tf.strings.join([tf.strings.as_string(x), tf.strings.as_string(y)], separator=',')

    def get_lookup_table(self, keys, vals):
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)

    def reset(self, validation=False, testing=False):
        self.time_step = tf.constant(0)
        self.cumulative_requests_count = tf.constant(0)
        
        # pick date and prepare trip data for the episode
        if testing:
            self.date = self.test_dates[0]
            self.test_dates.remove(self.date)
        elif validation:
            self.date = random.choice(self.remaining_validation_dates)
            self.remaining_validation_dates.remove(self.date)
        else:
            if not self.remaining_training_dates:
                self.remaining_training_dates = copy.deepcopy(self.training_dates)
            self.date = random.choice(self.remaining_training_dates)
            self.remaining_training_dates.remove(self.date)
        
        self.trip_data = pd.read_csv(self.data_dir + f'/trip_data/trips_{self.date}.csv', index_col=0)
        self.trip_data = self.trip_data.groupby(pd.cut(self.trip_data.pickup_time, np.arange(-1, self.episode_length, self.dt), labels=False)) # group per time step
        
        # initiate state
        requests_state = self.get_requests_state_tensor()

        if validation or testing:
            no_veh_per_zone_all_zones = int(self.n_veh/self.nodes_count)
            remaining_vehicles = self.n_veh % self.nodes_count
            v = no_veh_per_zone_all_zones * [i for i in range(self.nodes_count)] + [i for i in range(remaining_vehicles)]
        else:
            v = [random.randint(0, self.nodes_count-1) for i in range(self.n_veh)]
        tau = [0 for i in range(self.n_veh)]
        o1 = o2 = d1 = d2 = omega1 = omega2 = [-1 for i in range(self.n_veh)]        
        self.hvs = np.array([i for i in zip(v, tau, omega1, o1, d1, omega2, o2, d2)])
        vehicles_state = self.get_vehicles_state_tensor(tf.constant(self.hvs, tf.int32))
    
        misc_state = self.get_misc_state_tensor(vehicles_state, self.time_step, self.cumulative_requests_count)

        state = {"requests_state": requests_state,
                 "vehicles_state": vehicles_state,
                 "misc_state": misc_state}

        return state, tf.constant(self.hvs)

    def step(self, act):        
        self.time_step += 1
        
        try:
            requests_count = len(self.requests.index)
        except TypeError:
            requests_count = 0
                
        rew = np.zeros(self.n_req_max*self.n_veh)
        if requests_count > 0:
            rew = self.assign_accepted_requests(act, requests_count, rew)
        
        s_new = self.transition(tf.constant(self.hvs, tf.int32))
        self.hvs = s_new.numpy()

        next_requests_state = self.get_requests_state_tensor()
        next_vehicles_state = self.get_vehicles_state_tensor(s_new)
        next_misc_state = self.get_misc_state_tensor(next_vehicles_state, self.time_step, self.cumulative_requests_count)
        
        next_state = {"requests_state": next_requests_state,
                      "vehicles_state": next_vehicles_state,
                      "misc_state": next_misc_state}
        
        return next_state, tf.constant(rew, tf.float32), s_new
    
    def assign_accepted_requests(self, act, requests_count, rew):        
        for i in range(requests_count):
            veh_ix = act[i].numpy()
            if veh_ix != -1:
                o = self.requests.iloc[i,1]
                d = self.requests.iloc[i,2]
                
                # calculate time needed to pick up customer to check if request will be served within max waiting time
                time = self.hvs[veh_ix,1]
                if self.hvs[veh_ix,3] == -1:
                    if self.hvs[veh_ix,0] != o:
                        time += self.graph.loc[(self.hvs[veh_ix,0], o), 'travel_time']
                elif self.hvs[veh_ix,2] != -1:
                    if self.hvs[veh_ix,0] != self.hvs[veh_ix,3]:
                        time += self.graph.loc[(self.hvs[veh_ix,0], self.hvs[veh_ix,3]), 'travel_time']
                    time += self.graph.loc[(self.hvs[veh_ix,3], self.hvs[veh_ix,4]), 'travel_time']
                    if self.hvs[veh_ix,4] != o:
                        time += self.graph.loc[(self.hvs[veh_ix,4], o), 'travel_time']
                else:
                    if self.hvs[veh_ix,0] != self.hvs[veh_ix,4]:
                        time += self.graph.loc[(self.hvs[veh_ix,0], self.hvs[veh_ix,4]), 'travel_time']
                    if self.hvs[veh_ix,4] != o:
                        time += self.graph.loc[(self.hvs[veh_ix,4], o), 'travel_time']
                
                # reward: revenue
                if time <= self.max_waiting_time:
                    fare = self.graph.loc[(o,d),'fare']
                    rew[i*self.n_veh+veh_ix] += fare * self.local_rew_share 
                    rew[:] += fare * self.global_rew_share / (self.n_req_max*self.n_veh) 
                
                # reward: cost
                cost = self.graph.loc[(o,d),'distance'] # distance from o to d of request
                if self.hvs[veh_ix,3] == -1: # distance from current position to origin of new request if no other request assigned
                    if self.hvs[veh_ix,0] != o:
                        cost += self.graph.loc[(self.hvs[veh_ix,0], o), 'distance']
                else: # distance from destination of other request to origin of new request if other request assigned
                    if self.hvs[veh_ix,4] != o:
                        cost += self.graph.loc[(self.hvs[veh_ix,4], o), 'distance']
                cost *= self.cost_parameter
                
                rew[i*self.n_veh+veh_ix] += cost * self.local_rew_share 
                rew[:] += cost * self.global_rew_share / (self.n_req_max*self.n_veh) 
                
                # assign request to first position if it is empty, second position otherwise
                if self.hvs[veh_ix,3] == -1:
                    self.hvs[veh_ix,2] = 0 # omega1
                    self.hvs[veh_ix,3] = o # o1
                    self.hvs[veh_ix,4] = d # d1
                else:
                    self.hvs[veh_ix,5] = 0 # omega2
                    self.hvs[veh_ix,6] = o # o2
                    self.hvs[veh_ix,7] = d # d2

        return rew
    
    @tf.function
    def transition(self, s):
        target_node = tf.where(tf.reduce_all([s[:,2] == -1, s[:,3] != -1], axis=0), s[:,4], tf.where(s[:,3] != -1, s[:,3], s[:,0]))
        d = tf.where(s[:,0] == target_node, (s[:,0] + 1) % (self.nodes_count - 1), target_node)
        next_node = self.next_node_table[self.get_keys(s[:,0], d)]
        time_to_next_node = self.travel_time_table[self.get_keys(s[:,0], next_node)]
        
        cond_new_node = tf.reduce_all([s[:,1] == 0, s[:,0] != target_node], axis=0)
        cond_pickup = tf.reduce_all([s[:,3] != -1, s[:,2] != -1, s[:,0] == s[:,3], tf.reduce_any([s[:,1] == 1, s[:,1] == 0], axis=0)], axis=0)
        cond_dropoff = tf.reduce_all([s[:,3] != -1, s[:,2] == -1, s[:,0] == s[:,4], s[:,1] == 1], axis=0)
        cond_pickup_at_dropoff = tf.reduce_all([s[:,6] != -1, s[:,0] == s[:,6], s[:,1] == 1], axis=0)
        
        s_new_zero = tf.where(cond_new_node, next_node, s[:,0])
        s_new_one = tf.where(cond_new_node, time_to_next_node - 1, tf.where(s[:,1] > 0, s[:,1] - 1, s[:,1]))
        s_new_five = tf.where(s[:,6] != -1, s[:,5] + 1, s[:,5])
        s_new_two = tf.where(cond_pickup, -1, tf.where(cond_dropoff, tf.where(cond_pickup_at_dropoff, -1, s_new_five), tf.where(s[:,2] != -1, s[:,2] + 1, s[:,2])))
        s_new_three = tf.where(cond_dropoff, s[:,6], s[:,3])
        s_new_four = tf.where(cond_dropoff, s[:,7], s[:,4])
        s_new_five = tf.where(cond_dropoff, -1, s_new_five)
        s_new_six = tf.where(cond_dropoff, -1, s[:,6])
        s_new_seven = tf.where(cond_dropoff, -1, s[:,7])
                
        return tf.stack((s_new_zero, s_new_one, s_new_two, s_new_three, s_new_four, s_new_five, s_new_six, s_new_seven), axis=1)

    def get_requests_state_tensor(self):
        try:
            self.requests = self.trip_data.get_group(self.time_step.numpy())
            self.cumulative_requests_count += len(self.requests.index)
            if len(self.requests.index) > self.n_req_max:
                self.requests = self.requests.iloc[:self.n_req_max,:]
            
            origin_horizontal_idx = tf.expand_dims(tf.constant(self.requests.pickup_horizontal_idx/self.max_horizontal_idx, tf.float32), axis=1)
            origin_vertical_idx = tf.expand_dims(tf.constant(self.requests.pickup_vertical_idx/self.max_vertical_idx, tf.float32), axis=1)
            destination_horizontal_idx = tf.expand_dims(tf.constant(self.requests.dropoff_horizontal_idx/self.max_horizontal_idx, tf.float32), axis=1)
            destination_vertical_idx = tf.expand_dims(tf.constant(self.requests.dropoff_vertical_idx/self.max_vertical_idx, tf.float32), axis=1)
            distance = tf.expand_dims(tf.constant(self.requests.distance/self.max_distance, tf.float32), axis=1)
            requests_tensor = tf.concat([origin_horizontal_idx,origin_vertical_idx,destination_horizontal_idx,destination_vertical_idx,distance], axis=1)
            paddings = tf.constant([[0, self.n_req_max - tf.shape(requests_tensor)[0].numpy()], [0, 0]])
            requests_tensor = tf.pad(requests_tensor, paddings, constant_values=0.)
        
        except KeyError: # if no requests for current time step
            self.requests = []
            requests_tensor = tf.zeros([self.n_req_max, 5])
        
        return requests_tensor
    
    @tf.function
    def get_vehicles_state_tensor(self, s):
        position = tf.where(s[:,7] != -1, s[:,7], tf.where(s[:,4] != -1, s[:,4], s[:,0]))
        position_horizontal_idx = tf.expand_dims(tf.cast(self.horizontal_idx_table[position] / self.max_horizontal_idx, tf.float32), axis=1)
        position_vertical_idx = tf.expand_dims(tf.cast(self.vertical_idx_table[position] / self.max_vertical_idx, tf.float32), axis=1)
        
        # steps to position
        # always: tau, if first request exists + ...
        #  ... if already picked up: time from node to destination
        #  ... otherwise: time from node to origin + time from origin to destination
        #  if second request exists:
        #    + time from destination of first request to origin of second request
        #    + time from origin to destination of second request
        steps_to_position = s[:,1]
        o = tf.where(s[:,3] == -1, 0, s[:,3])
        d = tf.where(s[:,4] == -1, 1, s[:,4])
        idx1 = tf.where(d == s[:,0], (s[:,0] + 1) % (self.nodes_count - 1), d)
        idx2 = tf.where(o == s[:,0], (s[:,0] + 1) % (self.nodes_count - 1), o)
        steps_to_position += tf.where(s[:,3] != -1,
                                      tf.where(s[:,2] == -1,
                                               tf.where(s[:,0] != s[:,4], self.travel_time_table[self.get_keys(s[:,0], idx1)], 0),
                                               tf.where(s[:,0] != s[:,3], self.travel_time_table[self.get_keys(s[:,0], idx2)], 0) + self.travel_time_table[self.get_keys(o, d)]),
                                      0)
        o2 = tf.where(s[:,6] == -1, 0, s[:,6])
        d2 = tf.where(s[:,7] == -1, 1, s[:,7])
        idx3 = tf.where(d == o2, (o2 + 1) % (self.nodes_count - 1), d)
        steps_to_position += tf.where(s[:,6] != -1,
                                      tf.where(s[:,4] != s[:,6], self.travel_time_table[self.get_keys(idx3, o2)], 0) + self.travel_time_table[self.get_keys(o2, d2)],
                                      0)
        steps_to_position = steps_to_position / self.max_time
        steps_to_position = tf.expand_dims(tf.cast(steps_to_position, tf.float32), axis=1)
        
        count_assigned_requests = tf.expand_dims(tf.where(s[:,7] != -1, 1., tf.where(s[:,4] != -1, 0.5, 0.)), axis=1)
        
        return tf.concat([position_horizontal_idx, position_vertical_idx, steps_to_position, count_assigned_requests], axis=1)

    @tf.function
    def get_misc_state_tensor(self, vehicles_state, time_step, cumulative_requests_count):
        t = tf.cast(time_step / self.time_steps_count, tf.float32)
        
        if time_step == self.time_steps_count:
            r = tf.cast(cumulative_requests_count, tf.float32) / self.avg_request_count[time_step-1]
        else:
            r = tf.cast(cumulative_requests_count, tf.float32) / self.avg_request_count[time_step]
        
        tp = tf.reduce_sum(vehicles_state[:,2]) / (self.n_veh * 4)
        
        return tf.stack([t,r,tp])

    # compute time it takes until request will be picked up for each request/vehicle combination if it is assigned to the respective vehicle 
    # and based on this, get flag if request would be served within the maximum waiting time (used by actor and critic to compute additional feature)
    @tf.function
    def get_flag_within_max_waiting_time(self, vehicles_state, hvs, requests_state):
        hvs = tf.cast(hvs, tf.int32)
        
        time_to_location = tf.cast(tf.math.round(vehicles_state[:,:,2] * self.max_time), tf.int32)
        time_to_location = tf.tile(time_to_location, multiples=tf.constant([1,self.n_req_max]))
        
        location = tf.where(hvs[:,:,7] != -1, hvs[:,:,7], tf.where(hvs[:,:,4] != -1, hvs[:,:,4], hvs[:,:,0]))
        location = tf.tile(location, multiples=tf.constant([1,self.n_req_max]))
        
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,:,0] * self.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,:,1] * self.max_vertical_idx), tf.int32)
        pickup = self.zone_mapping_table[self.get_keys(pickup_horizontal, pickup_vertical)]
        
        pickup = tf.repeat(pickup, repeats=self.n_veh, axis=1)
        idx = tf.where(location == pickup, (pickup + 1) % (self.nodes_count - 1), location)
        time_location_to_pickup = tf.where(location != pickup, self.travel_time_table[self.get_keys(idx, pickup)], 0)
        
        time_to_pickup = time_to_location + time_location_to_pickup
        within_max_waiting_time = tf.cast(time_to_pickup <= self.max_waiting_time, tf.float16)
        return tf.expand_dims(within_max_waiting_time, axis=2)
