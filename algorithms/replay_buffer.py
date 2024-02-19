import tensorflow as tf


class ReplayBuffer(object):
    def __init__(self, size, normalized_rews, n_veh, n_req_max):
        self.normalized_rews = normalized_rews
        
        self.requests_states = tf.Variable(tf.zeros((size,) + (n_req_max,) + (5,), dtype=tf.float32))
        self.vehicles_states = tf.Variable(tf.zeros((size,) + (n_veh,) + (4,), dtype=tf.float32))
        self.misc_states = tf.Variable(tf.zeros((size,) + (3,), dtype=tf.float32))
        self.hvses = tf.Variable(tf.zeros((size,) + (n_veh,) + (8,), dtype=tf.int32))
        self.acts = tf.Variable(tf.zeros((size,) + (n_req_max,), dtype=tf.int32))
        self.rews = tf.Variable(tf.zeros((size,) + (n_veh*n_req_max,), dtype=tf.float32))
        self.next_requests_states = tf.Variable(tf.zeros((size,) + (n_req_max,) + (5,), dtype=tf.float32))
        self.next_vehicles_states = tf.Variable(tf.zeros((size,) + (n_veh,) + (4,), dtype=tf.float32))
        self.next_misc_states = tf.Variable(tf.zeros((size,) + (3,), dtype=tf.float32))
        self.next_hvses = tf.Variable(tf.zeros((size,) + (n_veh,) + (8,), dtype=tf.int32))
        
        if self.normalized_rews:
            self.masks = tf.Variable(tf.zeros((size,) + (n_veh*n_req_max,), dtype=tf.float32))
        
        self.maxsize = size
        self.size = tf.Variable(0, dtype=tf.int32)
        self.next_idx = tf.Variable(0, dtype=tf.int32)

    @tf.function
    def add(self, obs, hvs, act, rew, next_obs, next_hvs, mask=None):
        self.requests_states.scatter_nd_update([[self.next_idx]], [obs["requests_state"]])
        self.vehicles_states.scatter_nd_update([[self.next_idx]], [obs["vehicles_state"]])
        self.misc_states.scatter_nd_update([[self.next_idx]], [obs["misc_state"]])
        self.hvses.scatter_nd_update([[self.next_idx]], [tf.cast(hvs, tf.int32)])
        self.acts.scatter_nd_update([[self.next_idx]], [act])
        self.rews.scatter_nd_update([[self.next_idx]], [rew])
        self.next_requests_states.scatter_nd_update([[self.next_idx]], [next_obs["requests_state"]])
        self.next_vehicles_states.scatter_nd_update([[self.next_idx]], [next_obs["vehicles_state"]])
        self.next_misc_states.scatter_nd_update([[self.next_idx]], [next_obs["misc_state"]])
        self.next_hvses.scatter_nd_update([[self.next_idx]], [tf.cast(next_hvs, tf.int32)])
        
        if self.normalized_rews:
            self.masks.scatter_nd_update([[self.next_idx]], [mask])
        
        self.size.assign(tf.math.minimum(self.size + 1, self.maxsize))
        self.next_idx.assign((self.next_idx + 1) % self.maxsize)

    @tf.function
    def sample(self, batch_size):
        idxes = tf.random.uniform((batch_size,), maxval=self.size, dtype=tf.int32)
        
        obses_req = tf.gather(self.requests_states, idxes)
        obses_veh = tf.gather(self.vehicles_states, idxes)
        obses_misc = tf.gather(self.misc_states, idxes)
        hvses = tf.gather(self.hvses, idxes)
        acts = tf.gather(self.acts, idxes)
        rews = tf.gather(self.rews, idxes)
        next_obses_req = tf.gather(self.next_requests_states, idxes)
        next_obses_veh = tf.gather(self.next_vehicles_states, idxes)
        next_obses_misc = tf.gather(self.next_misc_states, idxes)
        next_hvses = tf.gather(self.next_hvses, idxes)
        
        if self.normalized_rews:
            flat_rews = tf.reshape(self.rews, [-1])
            flat_masks = tf.reshape(tf.cast(self.masks, dtype=tf.bool), [-1])
            gather_indices = tf.where(flat_masks)
            check = (tf.rank(gather_indices) == 2)
            tf.debugging.Assert(check, [tf.shape(gather_indices)])
            masked_rews = tf.gather(flat_rews, tf.squeeze(gather_indices, [1]))
            std = tf.math.reduce_std(masked_rews)
            rews /= std
        
        obses = {"requests_state": obses_req,
                 "vehicles_state": obses_veh,
                 "misc_state": obses_misc}
        next_obses = {"requests_state": next_obses_req,
                      "vehicles_state": next_obses_veh,
                      "misc_state": next_obses_misc}

        return obses, hvses, acts, rews, next_obses, next_hvses
