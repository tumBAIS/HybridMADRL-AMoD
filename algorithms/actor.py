"""Multi-agent actor including post-processing. All computations are made across a mini-batch and agents in parallel."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.regularizers import L2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


class Actor(tf.keras.Model):
    def __init__(self, args, env):
        super().__init__(name="Actor")
        
        self.request_embedding = RequestEmbedding(args)
        self.vehicle_embedding = VehicleEmbedding(args)
        self.requests_context = RequestsContext(args)
        self.vehicles_context = VehiclesContext(args)
        
        reg_coef = args["regularization_coefficient"]
        inner_layers = []
        for layer_size in args["inner_units"]:
            layer = Dense(layer_size, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(reg_coef))
            inner_layers.append(layer)
        self.inner_layers = inner_layers

        self.output_logits = Dense(2, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
        self.activation = Activation('softmax', dtype='float32')
        
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.post_processing_mode = args["post_processing"]
        
        self.env = env
    
    def call(self, state, hvs, test, request_masks):
        probs = self.compute_prob(state, hvs, request_masks["s"])
        act = self.post_process(probs, test, hvs, request_masks["l"])
        return tf.squeeze(act, axis=[0])

    def get_random_action(self, state, hvs, request_mask_l):
        probs = tf.ones((1, self.n_veh*self.n_req_max, 2)) / 2
        act = self.post_process(probs, tf.constant(False), tf.expand_dims(hvs, axis=0), request_mask_l)
        return tf.squeeze(act, axis=[0])

    @tf.function
    def compute_prob(self, state, hvs, request_mask_s):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]

        request_embedding = self.request_embedding(requests_state)
        vehicle_embedding = self.vehicle_embedding(vehicles_state)
        requests_context = self.requests_context(request_embedding, request_mask_s)
        vehicles_context = self.vehicles_context(vehicle_embedding)
        
        context = tf.concat([requests_context, vehicles_context], axis=1)
        context = tf.repeat(tf.expand_dims(context,axis=1), repeats=self.n_veh, axis=1)
        misc_state = tf.repeat(tf.expand_dims(tf.cast(state["misc_state"], tf.float16), axis=1), repeats=self.n_veh, axis=1)
        combined_input = tf.concat([misc_state, context, vehicle_embedding], axis=2)
        combined_input = tf.tile(combined_input, multiples=[1,self.n_req_max,1])
        
        request_embedding = tf.repeat(request_embedding, repeats=self.n_veh, axis=1)
        
        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        
        features = tf.concat([combined_input, within_max_waiting_time, request_embedding], axis=2)
        
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(features, paddings, constant_values=0.)
        
        for layer in self.inner_layers:
            features = layer(features)
                
        return self.activation(self.output_logits(features))

    def post_process(self, probs, test, hvs, request_mask_l):
        batch_size = tf.shape(probs)[0]
        
        probs = self.mask_probs(probs, hvs, request_mask_l)
        
        if self.post_processing_mode == "matching":
            act = self.get_action_from_probs(probs, test)
            act = self.reshape_transpose(act, batch_size, self.n_veh)
            act = act.numpy()
            action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.matching)(act[i,:,:]) for i in range(batch_size))
            act = tf.constant(action_list)
        
        if self.post_processing_mode == "weighted_matching":
            act = probs[:,:,1]
            sampled_action = self.get_action_from_probs(probs, test)
            act = act * tf.cast(sampled_action, tf.float32) # set score to zero if decision is reject
            act = self.reshape_transpose(act, batch_size, self.n_veh)
            act = act.numpy()
            action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.weighted_matching)(act[i,:,:]) for i in range(batch_size))
            act = tf.constant(action_list)
    
        return act
    
    @tf.function
    def mask_probs(self, probs, hvs, request_mask_l):    
        mask = hvs[:,:,6] == -1
        mask = tf.tile(mask, multiples=tf.constant([1,self.n_req_max]))
        mask = mask & tf.cast(request_mask_l, tf.bool)
        mask = tf.expand_dims(mask, axis=2)
        dummy_mask = tf.ones((hvs.shape[0], self.n_veh*self.n_req_max, 1), dtype=tf.bool)
        mask = tf.concat([dummy_mask, mask], axis=2)
        
        probs = probs * tf.cast(mask, tf.float32)
        probs /= tf.reduce_sum(probs, axis=2, keepdims=True)
        
        return probs
    
    @tf.function
    def get_action_from_probs(self, probs, test):
        if test:
            return tf.argmax(probs, axis=2, output_type=tf.int32)
        else:
            return tfp.distributions.Categorical(probs=probs).sample()
    
    @tf.function
    def reshape_transpose(self, act, batch_size, n_veh):
        act = tf.reshape(act, [batch_size, self.n_req_max, n_veh])
        return tf.transpose(act, perm=[0,2,1])
    
    def matching(self, x):
        return maximum_bipartite_matching(csr_matrix(x))
    
    def weighted_matching(self, x):
        matched_veh, matched_req = linear_sum_assignment(x, maximize=True) # weighted matching
        
        matched_weights = x[matched_veh, matched_req]
        matched_veh = np.where(matched_weights == 0., -1, matched_veh) # if weight is zero, correct matching decision to reject decision
        
        action = -np.ones(self.n_req_max, int)
        action[matched_req] = matched_veh
        
        return action


class RequestEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestEmbedding")
        
        self.embedding_layer = Dense(args["req_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, requests_state):
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(requests_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class VehicleEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehicleEmbedding")
        
        self.embedding_layer = Dense(args["veh_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, vehicles_state):
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(vehicles_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class RequestsContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestsContext")

        self.attention = args["attention"]
        reg_coef = args["regularization_coefficient"]

        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["req_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, requests_embeddings, request_mask_s):
        if self.attention:
            betas = Multiply()([self.w(self.W(requests_embeddings)), tf.expand_dims(request_mask_s, axis=2)])
        else:
            betas = tf.expand_dims(tf.cast(request_mask_s, tf.float16), axis=2)
        
        return tf.reduce_sum(betas * requests_embeddings, axis=1) / tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True)


class VehiclesContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehiclesContext")
        
        self.attention = args["attention"]
        reg_coef = args["regularization_coefficient"]

        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["veh_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, vehicles_embeddings):
        if self.attention:
            betas = self.w(self.W(vehicles_embeddings))
            return tf.reduce_mean(betas * vehicles_embeddings, axis=1)
        else:
            return tf.reduce_mean(vehicles_embeddings, axis=1)
