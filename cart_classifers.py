from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

class Classifer():

    def __init__(self, clf):
        self.clf = clf
        self.model = self.init_model()
        self.train_obs, self.train_actions, self.test_obs, self.test_actions = self.load_npy()
        self.train_model()
        print("{0} has accuracy: {1}%".format(self.clf, self.get_accuracy()))
        
    
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
        return action[0]
    
    def get_accuracy(self):
        test_preds = self.model.predict(self.test_obs)
        acc = accuracy_score(self.test_actions, test_preds)
        return round(acc, 4)*100

class SVM_Agent(Classifer):
    
    def __init__(self):
        super().__init__("SVM")

    def init_model(self):
        return svm.SVC()

class NB_Agent(Classifer):
    
    def __init__(self):
        super().__init__("NB")

    def init_model(self):
        return GaussianNB()

class DT_Agent(Classifer):
    
    def __init__(self):
        super().__init__("DT")

    def init_model(self):
        return tree.DecisionTreeClassifier()




        