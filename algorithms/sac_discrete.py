"""Set up networks and define one training iteration"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from actor import Actor
from critic import Critic


class SACDiscrete(tf.keras.Model):
    def __init__(self, args, env):
        super().__init__()
        
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.batch_size = args["batch_size"]
        self.alpha = tf.exp(tf.constant(args["log_alpha"]))
        self.tau = tf.constant(args["tau"])
        self.huber_delta = tf.constant(args["huber_delta"])
        self.gradient_clipping = tf.constant(args["gradient_clipping"])
        self.clip_norm = tf.constant(args["clip_norm"])
        self.discount = tf.Variable(args["discount"], dtype=tf.float32)

        self.actor = Actor(args, env)
        self.qf1 = Critic(args, env, name="qf1")
        self.qf2 = Critic(args, env, name="qf2")
        self.qf1_target = Critic(args, env, name="qf1_target")
        self.qf2_target = Critic(args, env, name="qf2_target")
        
        if args["scheduled_lr"]:
            init_lr = args["lr"]
            decay_steps = round((args["lr_decay_steps"] - args["min_steps"]) / args["update_interval"])
            end_lr = args["lr_end"]
            self.actor_optimizer = LossScaleOptimizer(Adam(PolynomialDecay(init_lr, decay_steps, end_lr)))
            self.qf1_optimizer = LossScaleOptimizer(Adam(PolynomialDecay(init_lr, decay_steps, end_lr)))
            self.qf2_optimizer = LossScaleOptimizer(Adam(PolynomialDecay(init_lr, decay_steps, end_lr)))
        else:
            lr = args["lr"]
            self.actor_optimizer = LossScaleOptimizer(Adam(lr))
            self.qf1_optimizer = LossScaleOptimizer(Adam(lr))
            self.qf2_optimizer = LossScaleOptimizer(Adam(lr))
        
        self.q1_update = tf.function(self.q_update)
        self.q2_update = tf.function(self.q_update)

    # get action from actor network for state input without batch dim
    def get_action(self, state, hvs, test=tf.constant(False)):
        state, request_masks = self.get_action_body(state)
        return self.actor(state, tf.expand_dims(hvs, axis=0), test, request_masks)
    
    @tf.function
    def get_action_body(self, state):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        misc_state = state["misc_state"]
        
        requests_state = tf.expand_dims(requests_state, axis=0)
        vehicles_state = tf.expand_dims(vehicles_state, axis=0)
        misc_state = tf.expand_dims(misc_state, axis=0)
        
        state = {"requests_state": requests_state,
                 "vehicles_state": vehicles_state,
                 "misc_state": misc_state}
                
        return state, self.get_masks(requests_state)

    # request masks of shape (batch size, n_req_max) and (batch size, n_req_max * n_veh)
    @tf.function
    def get_masks(self, requests_state):
        request_mask_s = tf.cast(tf.reduce_sum(requests_state, axis=2) > 0, tf.float32)
        request_mask_l = tf.repeat(request_mask_s, repeats=self.n_veh, axis=1)
        
        request_masks = {"s": tf.stop_gradient(request_mask_s),
                         "l": tf.stop_gradient(request_mask_l)}
        
        return request_masks

    # define one training iteration for a batch of experience
    def train(self, states, hvses, actions, rewards, next_states, next_hvses):
        request_masks = self.get_masks(states["requests_state"])
        
        cur_act_prob = self.actor.compute_prob(states, hvses, request_masks["s"])
        actions_current_policy = self.actor.post_process(cur_act_prob, tf.constant(False), hvses, request_masks["l"])
        
        target_q = self.target_Qs(rewards, next_states, request_masks, next_hvses)
        
        q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp = self.train_body(states, hvses, actions, target_q, request_masks, actions_current_policy)
        
        tf.summary.scalar(name="critic_loss", data=(q1_loss + q2_loss) / 2.)
        tf.summary.scalar(name="actor_loss", data=policy_loss)
        tf.summary.scalar(name="mean_ent", data=mean_ent)
        tf.summary.scalar(name="logp_mean", data=tf.reduce_mean(cur_act_logp))

    @tf.function
    def train_body(self, states, hvses, actions, target_q, request_masks, actions_current_policy):
        cur_q1 = self.qf1(states, actions_current_policy, hvses, request_masks["s"])
        cur_q2 = self.qf2(states, actions_current_policy, hvses, request_masks["s"])
        
        indices = tf.one_hot(actions, depth=self.n_veh, dtype=tf.int32) # indices that can be used to get Q(s,a) for correct a from Q(s), which is a vector with Q(s,a) for all possible a
        indices = tf.stop_gradient(tf.reshape(indices, shape=(self.batch_size, self.n_veh*self.n_req_max)))
        
        q1_loss = self.q1_update(states, hvses, actions, indices, target_q, self.qf1, self.qf1_optimizer, self.qf1_target, request_masks)
        q2_loss = self.q2_update(states, hvses, actions, indices, target_q, self.qf2, self.qf2_optimizer, self.qf2_target, request_masks)

        policy_loss, cur_act_prob, cur_act_logp = self.actor_update(states, hvses, request_masks, cur_q1, cur_q2)

        mean_ent = self.compute_mean_ent(cur_act_prob, cur_act_logp, request_masks["l"]) # mean entropy (info for summary output, not needed for algorithm)
        
        return q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp

    def target_Qs(self, rewards, next_states, request_masks, next_hvses):
        next_act_prob = self.actor.compute_prob(next_states, next_hvses, request_masks["s"])
        next_actions = self.actor.post_process(next_act_prob, tf.constant(False), next_hvses, request_masks["l"])
        return self.target_Qs_body(rewards, next_states, next_hvses, next_actions, next_act_prob, request_masks["s"])
    
    @tf.function
    def target_Qs_body(self, rewards, next_states, next_hvses, next_actions, next_act_prob, request_mask_s):
        next_q1_target = self.qf1_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q2_target = self.qf2_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q = tf.minimum(next_q1_target, next_q2_target)
        
        next_action_logp = tf.math.log(next_act_prob + 1e-8)
        target_q = tf.einsum('ijk,ijk->ij', next_act_prob, next_q - self.alpha * next_action_logp)        
        return tf.stop_gradient(rewards + self.discount * target_q)        

    def q_update(self, states, hvses, actions, indices, target_q, qf, qf_optimizer, qf_target, request_masks):
        with tf.GradientTape() as tape:
            cur_q = qf(states, actions, hvses, request_masks["s"]) # gives Q(s) for all a, not Q(s,a) for one a
            cur_q_selected = tf.gather_nd(cur_q, tf.expand_dims(indices, axis=2), batch_dims=2) # get correct Q(s,a) from Q(s)

            q_loss = self.huber_loss(target_q - cur_q_selected, self.huber_delta)
            q_loss = q_loss * request_masks["l"]
            q_loss = tf.reduce_mean(tf.reduce_sum(q_loss, axis=1)) # sum over agents and expectation over batch

            regularization_loss = tf.reduce_sum(qf.losses)
            scaled_q_loss = qf_optimizer.get_scaled_loss(q_loss + regularization_loss)
        
        scaled_gradients = tape.gradient(scaled_q_loss, qf.trainable_weights)
        gradients = qf_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        qf_optimizer.apply_gradients(zip(gradients, qf.trainable_weights))
        
        for target_var, source_var in zip(qf_target.weights, qf.weights):
            target_var.assign(self.tau * source_var + (1. - self.tau) * target_var)
        
        return q_loss

    @tf.function
    def huber_loss(self, x, delta):
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x) # MSE
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta) # linear
        return tf.where(tf.abs(x)<=delta, x=less_than_max, y=greater_than_max) # MSE for -delta < x < delta, linear otherwise

    @tf.function
    def actor_update(self, states, hvses, request_masks, cur_q1, cur_q2):
        with tf.GradientTape() as tape:
            cur_act_prob = self.actor.compute_prob(states, hvses, request_masks["s"])
            cur_act_logp = tf.math.log(cur_act_prob + 1e-8)
            
            policy_loss = tf.einsum('ijk,ijk->ij', cur_act_prob, self.alpha * cur_act_logp - tf.stop_gradient(tf.minimum(cur_q1, cur_q2)))
            policy_loss = policy_loss * request_masks["l"]
            policy_loss = tf.reduce_mean(tf.reduce_sum(policy_loss, axis=1)) # sum over agents and expectation over batch

            regularization_loss = tf.reduce_sum(self.actor.losses)
            scaled_loss = self.actor_optimizer.get_scaled_loss(policy_loss + regularization_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.actor.trainable_weights)
        gradients = self.actor_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))
        
        return policy_loss, cur_act_prob, cur_act_logp

    @tf.function
    def compute_mean_ent(self, cur_act_prob, cur_act_logp, request_mask_l):
        mean_ent = -tf.einsum('ijk,ijk->ij', cur_act_prob, cur_act_logp)
        mean_ent = mean_ent * request_mask_l
        return tf.reduce_sum(mean_ent) / tf.reduce_sum(request_mask_l) # mean over agents and batch
