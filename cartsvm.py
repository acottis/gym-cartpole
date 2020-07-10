from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

class SVM_Agent():

    def __init__(self):
        self.model = svm.SVC()
        self.train_obs, self.train_actions, self.test_obs, self.test_actions = self.load_npy()
        self.train_model()
        #self.predict_action(self.test_obs)
        print("Trained Neural network has accuracy: {0}%".format(self.get_accuracy()))
        
        

    def load_npy(self):
        actions = np.load('actions.npy')
        obs = np.load('obs.npy')

        split = int(len(actions)/10)

        train_actions = actions[:-split]
        test_actions = actions[-split:]

        train_obs = obs[:-split]
        test_obs = obs[-split:]

        return train_obs, train_actions, test_obs, test_actions

    def train_model(self):
        self.model.fit(self.train_obs, self.train_actions)

    def predict_action(self, obs):
        action = self.model.predict(obs)
        return action

    def get_accuracy(self):
        test_preds = self.model.predict(self.test_obs)
        acc = accuracy_score(self.test_actions, test_preds)
        return round(acc, 4)*100
        