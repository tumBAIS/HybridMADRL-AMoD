"""Estimate the request probability distribution per origin-destination-pair from the training data"""

import pandas as pd
import numpy as np


# set parameters
Laplace_smoothing = True  # bool if Laplace smoothing is used
alpha = 1  # smoothing parameter for Laplace smoothing

data_dir = "../data/data_11_small_zones"  # relative path to directory where data is stored
num_zones = 11  # number of zones in the considered instance

episode_length = 3600  # length of episode (in seconds)
dist_interval = 900  # length of time interval for which one distribution is computed (in seconds)
interval_count = int(episode_length / dist_interval)

# get list of training dates
training_dates = pd.read_csv(data_dir + '/training_dates.csv')
training_dates = training_dates.training_dates.tolist()
training_dates_count = len(training_dates)

# compute request count per origin-destination-pair per interval for which one distribution is computed
count = np.zeros((interval_count, num_zones, num_zones))

for date in training_dates:
    trip_data = pd.read_csv(data_dir + f'/trip_data/trips_{date}.csv', header=0, index_col=0, usecols=[0,1,2,3])
    
    trip_data = trip_data.groupby(pd.cut(trip_data.pickup_time, np.arange(-1, episode_length, dist_interval), labels=False))
    
    for interval in range(interval_count):
        try:
            data = trip_data.get_group(interval)
            
            for i in data.index:
                o = data.loc[i, "pickup_ID"]
                d = data.loc[i, "dropoff_ID"]
                
                count[interval,o,d] += 1
        
        except KeyError:
            pass

# compute and save average request count per interval and request probability distribution per origin-destination-pair
avg_req_count = np.zeros(interval_count, int)

for interval in range(interval_count):
    s = np.sum(count[interval,:,:])
    avg_req_count[interval] = round(s / training_dates_count)
    
    if Laplace_smoothing:
        count[interval,:,:] += alpha
        divisor = s + alpha * num_zones * (num_zones - 1)
        count[interval,:,:] /= divisor
        for i in range(num_zones):
            count[interval, i, i] = 0
    else:
        count[interval,:,:] /= s

if Laplace_smoothing:
    np.save(data_dir + '/od_prob_dist_smoothed.npy', count)
else:
    np.save(data_dir + '/od_prob_dist.npy', count)

np.save(data_dir + '/avg_req_count.npy', avg_req_count)
