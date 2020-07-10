
import gym
import numpy as np
import keyboard
import time
import csv
import random

# from prettygraphs import plot


# print("Action Space: ",env.action_space)
# print("Obs Space: ",env.observation_space)
# print("Obs Space High: ",env.observation_space.high)
# print("Obs Space Low: ",env.observation_space.low)

# obs = np.array([])


#ATTEMPTS = 100
BAD_ACTIONS = 19

env = gym.make("CartPole-v1")

def play(attempts):

    collected_obs = []
    collected_actions = []

    results = []

    for i in range(20):

        best_obs = []
        best_reward = 0
        best_actions = []

        for attempt in range(attempts):  
            observation = env.reset()
            done = False
            total_reward = 0
            obs = []

            actions = []
        #    print(attempt)
            # if not best_actions:
            #     actions = []
            # else:
            #     print(len(best_actions))
            #     actions = best_actions[: len(best_actions) - BAD_ACTIONS]
            #     #print(actions)

            while not done:

                #time.sleep(0.1)
                #env.render()

                # ACTION 0 = PUSH TO THE RIGHT, ACTION 1 =  PUSH TO THE LEFT

                #print(env.action_space.sample())

                # 50% 1 and 0 leads to 24 total reward
                #action = total_reward % 2

                if not best_actions:
                    action = random.randint(0,1)
                else:
                    if len(best_actions) > total_reward:
                        #print(len(best_actions), total_reward)
                        action = best_actions[total_reward]
                    else:
                        action = random.randint(0,1)
                #action = actions[total_reward]
                #print(total_reward)
                actions.append(action)
                obs.append(observation)
                
                observation, reward, done, info = env.step(action)
                
                total_reward += 1
                if done:
                #print("Finished with reward: ", total_reward)
                    if total_reward > best_reward:
                        best_reward = total_reward
                        if total_reward > 20:
                            best_actions = actions[:len(actions)-BAD_ACTIONS]
                            best_obs = obs[:len(actions)-BAD_ACTIONS]
                            #best_actions = actions[:int(len(actions)*0.25)]

        results.append(best_reward)
        print(best_reward)
        
        collected_actions += best_actions
        collected_obs += best_obs
        
    write_to_npy(collected_obs, collected_actions)
    env.close()
    average = sum(results) / len(results)
    return average

def write_to_npy(obs, actions):
    np.save('obs.npy', obs)
    np.save('actions.npy', actions)


# def drawgraph():
#     attempts = np.array([])
#     averages = np.array([])
#     for n in range(10,10001,10):
#         averages = np.append(averages, play(n))
#         attempts = np.append(attempts, n)
#         print(n)
#     #print(averages, attempts)
#     plot(attempts, averages)
# #drawgraph()

x = play(650)
print(x)

#print(averages)


            

