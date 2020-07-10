import numpy as np
import tensorflow as tf
from tensorflow import keras

# Global vars
EPOCH = 10

class NN_Agent():


    def __init__(self):

        self.epoch = EPOCH
        self.train_obs, self.train_actions, self.test_obs, self.test_actions = self.load_npy() 
        self.model = self.train_model(train_features=self.train_obs, train_labels=self.train_actions)
        print("Trained Neural network has accuracy: {0}%".format(self.get_accuracy()))
        
    def load_npy(self):
        actions = np.load('actions.npy')
        obs = np.load('obs.npy')

        split = int(len(actions)/20)

        train_actions = actions[:-split]
        test_actions = actions[-split:]

        train_obs = obs[:-split]
        test_obs = obs[-split:]

        return train_obs, train_actions, test_obs, test_actions

    def train_model(self, train_features, train_labels):  
        model = keras.Sequential([
                            keras.layers.Flatten(),
                            keras.layers.Dense(80, activation=tf.nn.relu),
                            keras.layers.Dense(2, activation=tf.nn.softmax)])
        model.compile(
                    optimizer=tf.optimizers.Adam(), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

        model.fit(train_features, train_labels, epochs=self.epoch, callbacks=myCallback())

        return model

    def predict_action(self, observation):
        pred = self.model.predict(observation)
        action = np.argmax(pred)
        return action

    def get_accuracy(self):
        acc = self.model.evaluate(self.test_obs, self.test_actions)
        return round(acc[1], 4)*100

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')>0.38):
    #   print("\nReached goal stopping neural net!")
    #   self.model.stop_training = True
      pass