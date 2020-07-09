import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def load_csv():
    np.set_printoptions(linewidth=200)
    data= np.loadtxt('inputs.csv',delimiter=",", dtype='str')
    data = np.transpose(data)

    obs, actions = data
    cleaned_obs = []

    for o in obs:
        o = o.replace('[','')
        o = o.replace(']','')
        o = o.split()
        cleaned_obs.append(o)

    train_features = np.array(cleaned_obs, dtype='float')
    train_labels = np.array(actions, dtype='int')

    return train_features, train_labels

def train_model(train_features, train_labels, epoch):  
    model = keras.Sequential([
                        keras.layers.Flatten(),
                        keras.layers.Dense(80, activation=tf.nn.relu),
                        keras.layers.Dense(2, activation=tf.nn.softmax)])
    model.compile(
                optimizer=tf.optimizers.Adam(), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=epoch)

    #test_features = np.array([[0.04775006, -0.55116988, -0.17562298,0.17487943]])
    return model

def predict_action(model, observation):
    pred = model.predict(observation)
    action = np.argmax(pred)
    return action

# train_features, train_labels = load_csv()
# model = train_model(train_features, train_labels)
# predict(model, np.array([[0.04775006, -0.55116988, -0.17562298,0.17487943]]))


