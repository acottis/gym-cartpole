import gym
import numpy as np
from cartnn import NN_Agent
from cartsvm import SVM_Agent
import time
import sys

ATTEMPTS = 10

# PICK CLASSIFIER
#CLASSIFER = "SVM"
CLASSIFER = "NN"

class play_cart(object):

    def __init__(self, clf):
        self.clf = clf
        if self.clf == "SVM":
            self.agent = SVM_Agent()
        elif self.clf == "NN":
            self.agent = NN_Agent()
        else:
            print("{0} is not a valid classifier selected".format(self.clf))
            sys.exit()
        self.play()

    def choose_action(self, observation):
        if self.clf == "SVM":
            action = self.agent.predict_action([observation])
            return action[0]
        if self.clf == "NN":
            obs = np.reshape(observation, [1,4])
            action = self.agent.predict_action(obs)
            return action

    def play(self):
        env = gym.make('CartPole-v0')
        for attempt in range(ATTEMPTS):  
            env.reset()
            done = False
            total_reward = 0
            action = 0

            while not done:
                env.render()
                #time.sleep(0.1)
                obs, reward, done,_ = env.step(action)
                action = self.choose_action(obs)
                #print(action)
                total_reward += reward

                if done:
                    print(total_reward)
        env.close()



if __name__ == "__main__":
    play_cart(CLASSIFER)

