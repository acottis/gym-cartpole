import gym
import numpy as np
from cartnn import NN_Agent
from cart_classifers import SVM_Agent, NB_Agent, DT_Agent, Ensemble_Agent
from cart_rng_agent import Random_Agent
import time
import sys

ATTEMPTS = 10

## PICK CLASSIFIER
CLASSIFER = "SVM" # Support Vector Machines
#CLASSIFER = "NN" # Neural Network
#CLASSIFER = "NB" # Naive Bayes
#CLASSIFER = "DT" # Decision Tree
#CLASSIFER = "Ensemble"

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
            self.agent = Ensemble_Agent()
        else:
            print("{0} is not a valid classifier selected".format(self.clf))
            sys.exit()
        self.play()

    def choose_action(self, observation):
        if self.clf == "SVM" or self.clf == "NB" or self.clf == "DT" or self.clf == "Ensemble":
            action = self.agent.predict_action([observation])
            return action
        elif self.clf == "NN":
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

def get_accuracy():
    NN_Agent().get_accuracy()
    SVM_Agent().get_accuracy()
    DT_Agent().get_accuracy()
    NB_Agent().get_accuracy()
    Ensemble_Agent().get_accuracy()

if __name__ == "__main__":
    ## Generates seed Data
    Random_Agent()

    ## Shows all accuracies
    get_accuracy()

    ## Starts game
    play_cart(CLASSIFER)
    


