
import gym
import numpy as np
import random

ATTEMPTS = 600
AGENT_RESETS = 10
BAD_ACTIONS = 18

class Random_Agent():

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.print_env()
        self.collected_obs = []
        self.collected_actions = []
        self.bad_actions = BAD_ACTIONS
        self.attempts = ATTEMPTS
        self.agent_resets = AGENT_RESETS

        self.play()
        self.env.close()
        self.write_to_npy(self.collected_obs, self.collected_actions)

    def print_env(self):

        print("Action Space: ",self.env.action_space)
        print("Obs Space: ",self.env.observation_space)
        print("Obs Space High: ",self.env.observation_space.high)
        print("Obs Space Low: ",self.env.observation_space.low)

    def play(self):

        for i in range(self.agent_resets):

            best_obs = []
            highest_reward = 0
            best_actions = []

            for attempt in range(self.attempts):  
                observation = self.env.reset()
                done = False
                current_loop_reward = 0
                obs = []
                actions = []

                while not done:

                    #self.env.render()

                    if not best_actions:
                        action = self.env.action_space.sample()
                    else:
                        if len(best_actions) > current_loop_reward:
                            action = best_actions[current_loop_reward]
                        else:
                            action = self.env.action_space.sample()
                    actions.append(action)
                    
                    obs.append(observation)

                    observation, reward, done, _ = self.env.step(action)


                    current_loop_reward += 1
                    if done:
                        if current_loop_reward > highest_reward:
                            highest_reward = current_loop_reward
                            if current_loop_reward > 20:
                                best_actions = actions[:len(actions)-self.bad_actions]
                                best_obs = obs[:len(actions)-self.bad_actions]

            #print("Highest reward this iteration:", highest_reward)
            self.collected_actions += best_actions
            self.collected_obs += best_obs
    
    def write_to_npy(self, obs, actions):
        np.save('obs.npy', obs)
        np.save('actions.npy', actions)

            

