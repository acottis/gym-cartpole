import gym
import numpy as np
from cartnn import load_csv, train_model, predict_action
import time

ATTEMPTS = 10
EPOCH = 5

def setup_nn():
    train_features, train_labels = load_csv()
    model = train_model(train_features, train_labels, epoch=EPOCH)
    return model

def choose_action(observation):
    obs = np.reshape(observation, [1,4])
    action = predict_action(model, obs)
    return action

def play_nn():
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
            action = choose_action(obs)
            #print(action)
            total_reward += reward

            if done:
                print(total_reward)
    env.close()

model = setup_nn()
play_nn()

