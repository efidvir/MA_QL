# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import gym
from gym import spaces
import cv2
#from google.colab.patches import cv2_imshow
#from google.colab import output
#import time
import os, sys
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import matplotlib.pyplot as plt
#plt.rcParams["figure.dpi"] = 300
from matplotlib import colors
#import networkx as nx
#from networkx.drawing.nx_agraph import write_dot
#from networkx.drawing.nx_pydot import write_dot

#import pygame
#from sklearn.preprocessing import normalize
#import graphviz
#from graphviz import Source
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
#import ffmpeg
#import moviepy.video.io.ImageSequenceClip
np.set_printoptions(precision=4)
from agents import Q_transmit_agent
from agents import AC_Agent
from env import transmit_env
from visualize import render
draw = render()

#agent type
#agent_type = 'Actor-Critic'
agent_type = 'Q_Learning'

#Global parameters
number_of_iterations = 1000000
force_policy_flag = True
number_of_agents = 3
np.random.seed(0)

#model
MAX_SILENT_TIME = 6
SILENT_THRESHOLD = 1
BATTERY_SIZE = 6
DISCHARGE = 1
MINIMAL_CHARGE = 1
CHARGE = 1
number_of_actions = 2

#learning params
GAMMA = 0.9
ALPHA = 0.01
#P_LOSS = 0
decay_rate = 0.999995

#for rendering
DATA_SIZE = 10

'''run realtime experiences'''
#T = [[] for i in range(number_of_agents)]
#for i in range(number_of_agents):
#    T[i] = np.zeros(shape=(BATTERY_SIZE * MAX_SILENT_TIME, MAX_SILENT_TIME * BATTERY_SIZE))  # transition matrix
policies = [[] for i in range(number_of_agents)]
values = [[] for i in range(number_of_agents)]
#pol_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))
#val_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))

occupied = 0
epsilon = np.ones(number_of_agents)
print(epsilon)
# initialize environment
env = [[] for i in range(number_of_agents)]
agent = [[] for i in range(number_of_agents)]
state = [[] for i in range(number_of_agents)]
actions = [[] for i in range(number_of_agents)]
transmit_or_wait_s = [[] for i in range(number_of_agents)]
score = [[] for i in range(number_of_agents)]
RAND = [[np.random.randint(10000)] for i in range(number_of_agents)]
rewards = [[] for i in range(number_of_agents)]
avg_rwrd = [[] for i in range(number_of_agents)]
for i in range(number_of_agents):
    #epsilon[i] = epsilon[i] -1/(number_of_agents+i)
    env[i] = transmit_env(BATTERY_SIZE, MAX_SILENT_TIME, SILENT_THRESHOLD, (1+i)*MINIMAL_CHARGE, (1+i)*DISCHARGE, CHARGE, DATA_SIZE, number_of_actions)
    if agent_type == 'Q_Learning':
        agent[i] = Q_transmit_agent(ALPHA, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions, (1+i)*MINIMAL_CHARGE,RAND[i])
        #Q_tables = [[] for i in range(number_of_iterations)]
    elif agent_type == 'Actor-Critic':
        agent[i] = AC_Agent(5*i*0.0000008, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions,MINIMAL_CHARGE)
        print('Make sure to adjust the learning rate')
    state[i] = env[i].initial_state
    actions[i] , transmit_or_wait_s[i] = agent[i].choose_action(state[i], epsilon[i])
    #policies[i] = agent[i].get_policy()
    #values[i] = agent[i].get_state_value(policies[i])


# plot reward function in use
#plt.plot(range(len(env[0].r_1)), env[0].r_1, 'o--', color='blue')
#plt.xticks(range(env[0].max_silence_time))
#plt.title('Reward function $r_1$')
#plt.show(block=False)
print(epsilon)
print('r_1 array: ', env[0].r_1)

errors = [[] for i in range(number_of_agents)]

for i in range(number_of_iterations):

    # all agents move a step and take a new action
    for j in range(number_of_agents):
        env[j].state = env[j].new_state

    if i % 200 == 0:
        j = np.random.randint(number_of_agents)
        current_energy, silent_time, priority = env[j].state
        if priority < 2:
            priority += 1
        env[j].state = [current_energy, silent_time, priority]

    # Gateway decision
    if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
        ack = 0
    elif sum(transmit_or_wait_s) == 1:
        ack = 1

    for j in range(number_of_agents):
        new_state, reward, occupied = env[j].time_step(actions[j], transmit_or_wait_s[j], sum(transmit_or_wait_s), ack)  # CHANNEL
        rewards[j] = reward

        env[j].new_state = new_state
        score[j].append(reward)

    for j in range(number_of_agents):
        np.random.seed(j)
        #print('Agent ', j)
        #draw.render_Q_diffs(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,i,env[j].state,actions[j], rewards[j], env[j].new_state)
        actions[j], transmit_or_wait_s[j] = agent[j].step(env[j].state, rewards[j], actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j])
        epsilon[j] = epsilon[j] * decay_rate

    if i % 100 == 0:
        print('step: ', i, '100 steps AVG mean score: ',np.mean(score[0][-100:-1]),epsilon[0])

'''
for j in range(number_of_agents):
    print('video done')
    draw.render_Q_diffs_video(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,number_of_iterations)
'''

print(epsilon)
# plt.plot(errors)
#video.release()


#Agent evaluation
# No exploration
epsilon = np.zeros(number_of_agents)

data = []
collisions = 0
agent_clean = [np.zeros(1) for i in range(number_of_agents)]
wasted = 0
priority_time=[np.zeros(1) for i in range(number_of_agents)]
num_of_eval_iner = 1000

#reset priority
for j in range(number_of_agents):
    current_energy, silent_time, priority = env[j].state
    env[j].state = [current_energy, silent_time, 0]

for i in range(num_of_eval_iner):
    for a in range(number_of_agents):
        env[a].state = env[a].new_state
        current_energy, silent_time, priority = env[a].state
        print('agent', a , 'priority ', priority)

    if i % 200 == 0:
        j = 2#np.random.randint(number_of_agents)
        current_energy, silent_time, priority = env[j].state
        if priority < 2:
            priority += 1
        env[j].state = [current_energy, silent_time, priority]


    # Gateway decision
    if sum(transmit_or_wait_s) > 1 or sum(transmit_or_wait_s) == 0:
        ack = 0
    elif sum(transmit_or_wait_s) == 1:
        ack = 1

    if sum(transmit_or_wait_s) > 1:
        collisions += 1
        data.append(1)
    if sum(transmit_or_wait_s) == 1:
        for a in range(number_of_agents):
            if transmit_or_wait_s[a] == 1:
                agent_clean[a] += 1
                data.append(a+2)
    if sum(transmit_or_wait_s) == 0:
        wasted += 1
        data.append(0)
    for a in range(number_of_agents):
        new_state, reward, occupied = env[a].time_step(actions[a],transmit_or_wait_s[a], sum(transmit_or_wait_s), ack)  # CHANNEL
        env[a].new_state = new_state
        actions[a] ,transmit_or_wait_s[a] = agent[a].step(env[a].state, reward, actions[a],transmit_or_wait_s[a], env[a].new_state, epsilon[a])

    for a in range(number_of_agents):
        # decompose state
        current_energy, slient_time, priority= env[a].state
        if priority > 0:
            priority_time[a] +=1
        # decompose new state
        next_energy, next_silence, next_priority= env[a].new_state

print('collisions', collisions)
for a in range(number_of_agents):
    print('agent{d}'.format(d=a), agent_clean[a] , 'rate:  ', env[a].discharge_rate)
    for i in range(int(len(score[a])/1000)):
        avg_rwrd[a].appent(np.mean(score[a][1000*i:1000*(i+1)]))
    plt.plot(range(len(avg_rwrd[a])), avg_rwrd[a])
    plt.legend(range(number_of_agents))
    plt.show(block=False)
print('wasted', wasted)

print('priority_time: ', priority_time)
print(data)
draw.last_1k_slots(data, number_of_agents)
'''
for i in range(number_of_agents):
    print('Agent ', i)
    print('\n')

    #draw.plot_Q_values(Q_tables,number_of_iterations)

for i in range(number_of_agents):
    print('Agent ',i,' Q table:', agent[i].Q[:, :, :])
    draw.render_Q(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j, i, env[j].state)
    cv2.waitKey(0)
'''