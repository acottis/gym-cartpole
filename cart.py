import gym
import numpy as np
from cartnn import Agent
import time

ATTEMPTS = 10
class play_cart():

    def __init__(self):
        self.nn_agent = self.setup_nn()
        self.play_nn()

    def setup_nn(self):
        agent = Agent()
        return agent

    def choose_action(self, observation):
        obs = np.reshape(observation, [1,4])
        action = self.nn_agent.predict_action(obs)
        return action

    def play_nn(self):
        env = gym.make('CartPole-v0')
        for attempt in range(ATTEMPTS):  
            env.reset()
            done = False
            total_reward = 0
            action = 0

            while not done:
                #env.render()
                #time.sleep(0.1)
                obs, reward, done,_ = env.step(action)
                action = self.choose_action(obs)
                #print(action)
                total_reward += reward

                if done:
                    print(total_reward)
        env.close()



if __name__ == "__main__":
    play_cart()

