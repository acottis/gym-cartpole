import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def load_npy():
    actions = np.load('actions.npy')
    obs = np.load('obs.npy')

    return obs, actions

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.65):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

def train_model(train_features, train_labels, epoch):  
    model = keras.Sequential([
                        keras.layers.Flatten(),
                        keras.layers.Dense(80, activation=tf.nn.relu),
                        keras.layers.Dense(2, activation=tf.nn.softmax)])
    model.compile(
                optimizer=tf.optimizers.Adam(), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=epoch, callbacks=myCallback())

    return model

def predict_action(model, observation):
    pred = model.predict(observation)
    action = np.argmax(pred)
    
    return action

# train_features, train_labels = load_csv()
# model = train_model(train_features, train_labels)
# predict(model, np.array([[0.04775006, -0.55116988, -0.17562298,0.17487943]]))
