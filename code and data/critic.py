"""Multi-agent critic: takes state and all agents' actions as input but ignores action of agent under consideration, computes Q-values 
   for all possible actions of this agent. All computations are made across a mini-batch and agents in parallel."""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.regularizers import L2


class Critic(tf.keras.Model):
    def __init__(self, args, env, name="Critic"):
        super().__init__(name=name)
        
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

        self.output_layer = Dense(2, activation=None, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
        self.output_cast = Activation('linear', dtype='float32')
        
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        
        self.env = env
    
    @tf.function
    def call(self, state, act, hvs, request_mask_s):        
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        
        # add action to request state: reject/assign (0/1)
        requests_input = tf.cast(tf.expand_dims(tf.where(act==-1, 0., 1.), axis=2), tf.float16)
        requests_input = tf.concat([requests_state, requests_input], axis=2)
        
        # add action to vehicle state: o and d of newly assigned request
        act = tf.one_hot(act, depth=self.n_veh, axis=-1, dtype=tf.float16)
        vehicles_input = tf.matmul(act, requests_state[:,:,:-1], transpose_a=True)
        vehicles_input = tf.concat([vehicles_state, vehicles_input], axis=2)

        request_embedding = self.request_embedding(requests_input)
        vehicle_embedding = self.vehicle_embedding(vehicles_input)
        requests_context = self.requests_context(request_embedding, request_mask_s)
        vehicles_context = self.vehicles_context(vehicle_embedding)

        requests_input = tf.concat([requests_context, requests_state], axis=2)
        requests_input = tf.repeat(requests_input, repeats=self.n_veh, axis=1)

        misc_state = tf.repeat(tf.expand_dims(state["misc_state"],axis=1), repeats=self.n_veh, axis=1)
        vehicles_input = tf.concat([misc_state, vehicles_context, vehicles_state], axis=2)
        vehicles_input = tf.tile(vehicles_input, multiples=[1,self.n_req_max,1])

        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        
        features = tf.concat([vehicles_input, within_max_waiting_time, requests_input], axis=2)
        
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(features, paddings, constant_values=0.)
        
        for layer in self.inner_layers:
            features = layer(features)
        
        return self.output_cast(self.output_layer(features))


class RequestEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestEmbedding")
        
        self.embedding_layer = Dense(args["req_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, requests_inputs):
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(requests_inputs, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class VehicleEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehicleEmbedding")
        
        self.embedding_layer = Dense(args["veh_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, vehicles_inputs):
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(vehicles_inputs, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class RequestsContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestsContext")

        self.n_req_max = args["max_req_count"]
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
        
        requests_context = tf.reduce_sum(betas * requests_embeddings, axis=1) / tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True)
        requests_context = tf.repeat(tf.expand_dims(requests_context, axis=1), repeats=self.n_req_max, axis=1)
        return requests_context - betas * requests_embeddings / tf.expand_dims(tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True), axis=2) # exclude contribution of individual requests


class VehiclesContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehiclesContext")

        self.n_veh = args["veh_count"]
        self.attention = args["attention"]
        reg_coef = args["regularization_coefficient"]
        self.batch_size = args["batch_size"]

        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["veh_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, vehicles_embeddings):        
        if self.attention:
            betas = self.w(self.W(vehicles_embeddings))
        else:
            betas = tf.ones([self.batch_size, self.n_veh, 1], tf.float16)
            
        vehicles_context = tf.reduce_mean(betas * vehicles_embeddings, axis=1)
        vehicles_context = tf.repeat(tf.expand_dims(vehicles_context, axis=1), repeats=self.n_veh, axis=1) 
        return vehicles_context - betas * vehicles_embeddings / tf.cast(self.n_veh, tf.float16) # exclude contribution of individual vehicles
