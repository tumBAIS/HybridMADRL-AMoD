"""Training loop incl. validation and testing"""

import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, policy, env, args):
        self.policy = policy
        self.env = env
        
        self.episode_max_steps = int(args["episode_length"]/args["time_step_size"]) # no. of steps per episode
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.max_steps = args["max_steps"]
        self.min_steps = args["min_steps"]
        self.random_steps = args["random_steps"]
        self.update_interval = args["update_interval"]
        self.validation_interval = args["validation_interval"]
        self.tracking_interval = args["tracking_interval"]
        self.profile_interval = args["profile_interval"]
        self.rb_size = args["rb_size"]        
        self.batch_size = args["batch_size"]
        self.scheduled_discount = args["scheduled_discount"]
        self.scheduled_discount_values = args["scheduled_discount_values"]
        self.scheduled_discount_steps = args["scheduled_discount_steps"]
        self.normalized_rews = args["normalized_rews"]
        self.data_dir = args["data_dir"]
        self.results_dir = args["results_dir"]
        self.validation_episodes = len(pd.read_csv(self.data_dir + '/validation_dates.csv').validation_dates.tolist())

        # save arguments and environment variables
        with open(self.results_dir + '/args.txt', 'w') as f: f.write(str(args))
        with open(self.results_dir + '/environ.txt', 'w') as f: f.write(str(dict(os.environ)))

        # initialize model saving and potentially restore saved model
        self.set_check_point(args["model_dir"])

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self.results_dir)
        self.writer.set_as_default()

    # initialize model saving and potentially restore saved model
    def set_check_point(self, model_dir):
        self.checkpoint = tf.train.Checkpoint(policy=self.policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.results_dir, max_to_keep=500)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self.checkpoint.restore(latest_path_ckpt)

    def __call__(self):   
        total_steps = 0
        episode_steps = 0
        episode_reward = 0.
        self.validation_rewards = []
        if self.scheduled_discount: discount_iter = 0
        profiling_start_step = self.profile_interval

        replay_buffer = ReplayBuffer(self.rb_size, self.normalized_rews, self.n_veh, self.n_req_max)

        state, hvs = self.env.reset()

        while total_steps < self.max_steps:
            if (total_steps + 1) % 100 == 0: tf.print("Started step", total_steps+1)
            
            if self.scheduled_discount:
                if self.scheduled_discount_steps[discount_iter] == total_steps:
                    self.policy.discount.assign(self.scheduled_discount_values[discount_iter])
                    discount_iter += 1
                    if discount_iter == len(self.scheduled_discount_values):
                        discount_iter = 0
            
            if (total_steps + 1) % self.profile_interval == 0:
                profiling_start_step = total_steps
                tf.profiler.experimental.start(self.results_dir)
            
            with tf.profiler.experimental.Trace('train', step_num=total_steps, _r=1):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                elif total_steps < self.random_steps:
                    request_masks = self.policy.get_masks(tf.expand_dims(state["requests_state"], axis=0))
                    action = self.policy.actor.get_random_action(state, hvs, request_masks["l"])
                else:
                    action = self.policy.get_action(state, hvs)
                
                next_state, reward, next_hvs = self.env.step(action)
    
                if ~tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    next_state_adapted = copy.deepcopy(next_state)
                    next_state_adapted["requests_state"] = state["requests_state"] # replace new requests with old requests (solution for dimensionality problem in critic loss)
                    if self.normalized_rews:
                        rew_mask = tf.squeeze(self.policy.get_masks(tf.expand_dims(state["requests_state"], axis=0))["l"], axis=0)                        
                        replay_buffer.add(obs=state, hvs=hvs, act=action, rew=reward, next_obs=next_state_adapted, next_hvs=next_hvs, mask=rew_mask)
                    else:
                        replay_buffer.add(obs=state, hvs=hvs, act=action, rew=reward, next_obs=next_state_adapted, next_hvs=next_hvs)
    
                state = next_state
                hvs = next_hvs
                total_steps += 1
                episode_steps += 1
                episode_reward += tf.reduce_sum(reward).numpy()
                
                tf.summary.experimental.set_step(total_steps)            
    
                if total_steps >= self.min_steps and total_steps % self.update_interval == 0:
                    states, hvses, acts, rews, next_states, next_hvses = replay_buffer.sample(self.batch_size)
                    with tf.summary.record_if(total_steps % self.tracking_interval == 0):
                        self.policy.train(states, hvses, acts, rews, next_states, next_hvses)

            if total_steps % self.validation_interval == 0:
                avg_validation_reward = self.validate_policy()
                tf.summary.scalar(name="avg_reward_per_validation_episode", data=avg_validation_reward)
                self.validation_rewards.append(avg_validation_reward)
                
                self.checkpoint_manager.save()

            if episode_steps == self.episode_max_steps:
                tf.summary.scalar(name="training_return", data=episode_reward)
                episode_steps = 0
                episode_reward = 0.
                state, hvs = self.env.reset()

            if total_steps == profiling_start_step + 5: tf.profiler.experimental.stop() # stop profiling 5 steps after start

        tf.summary.flush()
                
        self.test_policy()
        
        tf.print("Finished")

    # compute average reward per validation episode achieved by current policy
    def validate_policy(self):
        validation_reward = 0.
        
        for i in range(self.validation_episodes):
            state, hvs = self.env.reset(validation=True)
            
            for j in range(self.episode_max_steps):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                else:
                    action = self.policy.get_action(state, hvs, test=tf.constant(True))
                
                next_state, reward, hvs = self.env.step(action)
                
                validation_reward += tf.reduce_sum(reward).numpy()
                
                state = next_state
        
        avg_validation_reward = validation_reward / self.validation_episodes
        
        self.env.remaining_validation_dates = copy.deepcopy(self.env.validation_dates) # reset list of remaining validation dates

        return avg_validation_reward

    # compute rewards per test episode with best policy
    def test_policy(self):
        ckpt_id = np.argmax(self.validation_rewards) + 1
        self.checkpoint.restore(self.results_dir + f"/ckpt-{ckpt_id}")
        
        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv').test_dates.tolist()
        
        test_rewards = []
        
        for i in range(len(test_dates)):
            test_reward = 0.
            
            state, hvs = self.env.reset(testing=True)
            
            for j in range(self.episode_max_steps):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                else:
                    action = self.policy.get_action(state, hvs, test=tf.constant(True))
                
                next_state, reward, hvs = self.env.step(action)
                
                test_reward += tf.reduce_sum(reward).numpy()
                
                state = next_state
            
            test_rewards.append(test_reward)
                
        pd.DataFrame({"test_rewards_RL": test_rewards}, index=test_dates).to_csv(self.results_dir + "/test_rewards.csv")
        with open(self.results_dir + "/avg_test_reward.txt", 'w') as f: f.write(str(np.mean(test_rewards)))
