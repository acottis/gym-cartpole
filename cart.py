import gym
import numpy as np
from cartnn import NN_Agent
from cart_classifers import SVM_Agent, NB_Agent, DT_Agent
import time
import sys

ATTEMPTS = 10

## PICK CLASSIFIER
#CLASSIFER = "SVM" # Support Vector Machines
#CLASSIFER = "NN" # Neural Network
#CLASSIFER = "NB" # Naive Bayes
#CLASSIFER = "DT" # Decision Tree
CLASSIFER = "Ensemble"

class play_cart(object):

    def __init__(self, clf):
        self.clf = clf
        if self.clf == "SVM":
            self.agent = SVM_Agent()
        elif self.clf == "NB":
            self.agent = NB_Agent()
        elif self.clf == "NN":
            self.agent = NN_Agent()
        elif self.clf == "DT":
            self.agent = DT_Agent()
        elif self.clf == "Ensemble":
            self.agents = [NB_Agent(), DT_Agent(), SVM_Agent()]
        else:
            print("{0} is not a valid classifier selected".format(self.clf))
            sys.exit()
        self.play()

    def choose_action(self, observation):
        if self.clf == "SVM" or self.clf == "NB" or self.clf == "DT":
            action = self.agent.predict_action([observation])
            return action
        elif self.clf == "NN":
            obs = np.reshape(observation, [1,4])
            action = self.agent.predict_action(obs)
            return action
        if self.clf == "Ensemble":
            actions = []
            for a in self.agents:
                action = a.predict_action([observation])
                actions.append(action)
            # Finds the most common predicted action
            action = np.argmax(np.bincount(actions))
            return action

    def play(self):
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
    play_cart(CLASSIFER)

