import multiprocessing as mp
from multiprocessing import Process, Pool
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
import numpy as np
import pickle
import gym
from gym import spaces
import cv2
#from google.colab.patches import cv2_imshow
#from google.colab import output
import time
import os, sys
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
#plt.rcParams["figure.dpi"] = 300
from matplotlib import colors
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
number_of_iterations = 10000
number_of_agents = 3
np.random.seed(0)
force_policy_flag = True
#model
MAX_SILENT_TIME = 6
SILENT_THRESHOLD = 0
BATTERY_SIZE = 6
MAX_IDLE_TIME = 3
DISCHARGE = 2
MINIMAL_CHARGE = 2
CHARGE = 1
number_of_actions = 2

#learning params
GAMMA = 0.9
ALPHA = 0.1
alphas = [0.1,0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ps = [0.9,0.5,0.1]
#P_LOSS = 0
decay_rate = 0.999

#for rendering
DATA_SIZE = 10
# visualizationd flags
state_transition_graph = False
evaluation_slots = False
def run_simulation(number_of_iterations,decay_rate, number_of_agents,force_policy_flag,DISCHARGE, GAMMA, ALPHA):
    # Global parameters
    np.random.seed(0)
    # model
    MAX_SILENT_TIME = 2*number_of_agents
    SILENT_THRESHOLD = 0
    BATTERY_SIZE = 2*number_of_agents
    MAX_IDLE_TIME = number_of_agents
    MINIMAL_CHARGE = DISCHARGE
    CHARGE = 1
    number_of_actions = 2
    p=0.5
    # for rendering
    DATA_SIZE = 10
    '''run realtime experiences'''
    if state_transition_graph:
        T = [[] for i in range(number_of_agents)]
        for i in range(number_of_agents):
            T[i] = np.zeros(shape=(BATTERY_SIZE * MAX_SILENT_TIME * MAX_IDLE_TIME, MAX_SILENT_TIME * BATTERY_SIZE * MAX_IDLE_TIME))  # transition matrix
    else:
        T=[]
    policies = [[] for i in range(number_of_agents)]
    values = [[] for i in range(number_of_agents)]
    #pol_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))
    #val_t = np.ndarray(shape=(number_of_iterations, number_of_agents, BATTERY_SIZE, MAX_SILENT_TIME))

    occupied = 0
    epsilon = np.ones(number_of_agents)*0.5
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
    r_max = 0
    for i in range(number_of_agents):
        #epsilon[i] = epsilon[i] -1/(number_of_agents+i)
        env[i] = transmit_env(BATTERY_SIZE, MAX_SILENT_TIME, SILENT_THRESHOLD, MAX_IDLE_TIME, MINIMAL_CHARGE, DISCHARGE, CHARGE, DATA_SIZE, number_of_actions)
        if agent_type == 'Q_Learning':
            agent[i] = Q_transmit_agent(ALPHA, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME,MAX_IDLE_TIME, DATA_SIZE, number_of_actions, MINIMAL_CHARGE,RAND[i])
            #Q_tables = [[] for i in range(number_of_iterations)]
        elif agent_type == 'Actor-Critic':
            agent[i] = AC_Agent(5*i*0.0000008, GAMMA, BATTERY_SIZE, MAX_SILENT_TIME, DATA_SIZE, number_of_actions,MINIMAL_CHARGE)
            print('Make sure to adjust the learning rate')
        state[i] = env[i].initial_state
        actions[i] , transmit_or_wait_s[i] = agent[i].choose_action(state[i], epsilon[i],p)
        #policies[i] = agent[i].get_policy()
        #values[i] = agent[i].get_state_value(policies[i])


    # plot reward function in use
    #plt.plot(range(len(env[0].r_1)), env[0].r_1, 'o--', color='blue')
    #plt.xticks(range(env[0].max_silence_time))
    #plt.title('Reward function $r_1$')
    #plt.show(block=False)
    print('Final epsilon values:  ', epsilon)
    #print('r_1 array: ', env[0].r_1)
    bad_counter = np.zeros(number_of_agents)
    errors = [[] for i in range(number_of_agents)]
    resolution = 1
    #Qs = np.array([[np.array ([[agent[j].Q] for j in range(number_of_agents)])] for i in range(number_of_iterations)])
    for i in range(number_of_iterations):
        #Qs[i] = np.array ([[agent[j].Q] for j in range(number_of_agents)])

        # all agents move a step and take a new action
        for j in range(number_of_agents):
            env[j].state = env[j].new_state

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
            actions[j], transmit_or_wait_s[j] = agent[j].step(env[j].state, rewards[j], actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j],p)
            '''
            if epsilon[j] < 0.01:
                if (rewards[j] == 0 and actions[j] == 1):
                    bad_counter[j] += 1
                    if bad_counter[j] > 1000:
                        epsilon[j] += 0.1
                        bad_counter[j] = 0
            '''
            epsilon[j] = epsilon[j] * decay_rate
        if i % 1000 == 0:
            print('alpha: ',ALPHA, 'step: ', i, '100 steps AVG mean score: ',np.mean(score[0][-1000:-1]),epsilon[0])
            if np.mean(score[0][-1000:-1]) > r_max:
                r_max = np.mean(score[0][-1000:-1])

            #if r_max == 1.0:
            #    epsilon = np.zeros(number_of_agents)
                #break


    #draw.render_q_by_agent(Qs,number_of_agents)

    '''
    for j in range(number_of_agents):
        print('video done')
        draw.render_Q_diffs_video(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,number_of_iterations)
    '''

    print(epsilon)
    # plt.plot(errors)
    #video.release()
    for a in range(number_of_agents):
        for i in range(int(len(score[a]) / 1000)):
            avg_rwrd[a].append(np.mean(score[a][1000 * i:1000 * (i + 1)]))
    return avg_rwrd, T


def make_pool_parameters(iter):
    args = []
    for alpha in iter:
        args.append((number_of_iterations,decay_rate, number_of_agents,force_policy_flag,DISCHARGE, GAMMA, alpha))
    return args

hyper_sim_args = make_pool_parameters(alphas)


#create a matrix for each of the tuned hyperparameter value
mat = [[] for i in range(len(hyper_sim_args))]
max_mat = [[] for i in range(len(hyper_sim_args))]
min_mat = [[] for i in range(len(hyper_sim_args))]
avg_mat =  [[] for i in range(len(hyper_sim_args))]
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
#for sims in range(len(hyper_sim_args)):
avg_num=10
if __name__ == '__main__':
    for p in ps:
        mp.freeze_support()
        for var in range(avg_num):
            with Pool() as pool:
                for index, results in enumerate(pool.starmap(run_simulation, hyper_sim_args)):
                    avg_rwrd, T = results

                    if len(mat[index]) == 0:
                        mat[index] = np.array(avg_rwrd[0])
                        #max_mat[index] = np.array(avg_rwrd[0])
                        #min_mat[index] = np.array(avg_rwrd[0])
                        #avg_mat[index] = np.array(avg_rwrd[0])
                    else:
                        mat[index] = np.vstack((mat[index], np.array(avg_rwrd[0])))
                        # print(mat.shape(),avg_rwrd.shape())


            print('AVG #',var, '-----------------------------')

        #for tmp in range(len(hyper_sim_args)):
        #    mat[tmp] = np.delete(mat[tmp], 0, axis=0)
            #max_mat[tmp] = np.delete(max_mat[tmp], 0, axis=0)
            #min_mat[tmp] = np.delete(min_mat[tmp], 0, axis=0)
            #avg_mat[tmp] = np.delete(avg_mat[tmp], 0, axis=0)
        for i in range(len(hyper_sim_args)):
            max_mat[i] = np.amax(mat[i], axis=0)
            min_mat[i] = np.amin(mat[i], axis=0)
            avg_mat[i] = np.average(mat[i], axis=0)
    #surface_mat0 = np.vstack(avg_mat, axis=1)
        print(np.array(avg_mat).shape)
        X,Y = np.meshgrid(range(int(number_of_iterations/1000)),alphas)
        ax.plot_surface(X, Y, np.array(avg_mat), alpha=0.3, label='p = {d}'.format(d=p))
    print('P = ', p, '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    ax.set_zlabel('Average reward')
    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Number of iterations X1000')
    legend_elements = [Patch(facecolor='blue',label='p = 0.9'),Patch(facecolor='green',label='p = 0.5'),Patch(facecolor='orange',label='p = 0.1')]
    ax.legend(handles=legend_elements, loc='upper left')
    #plt.plot(range(len(avg_mat[0])), avg_mat[0])
    #plt.legend(range(number_of_agents))
    plt.show()

    pickle.dump(ax, open('FigureObject_test.fig.pickle', 'wb'))
#avg_rwrd, score, T, agent =
'''
#Agent evaluation
# No exploration
epsilon = np.zeros(number_of_agents)

data = []
collisions = 0
agent_clean = [np.zeros(1) for i in range(number_of_agents)]
wasted = 0
num_of_eval_iner = 1000
active =[]
decay_rate = 0.9999
re_explore = False

for i in range(num_of_eval_iner):
    #print(env[0].new_state,env[1].new_state,env[2].new_state)
    for a in range(number_of_agents):
        env[a].state = env[a].new_state
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

    for j in range(number_of_agents):
        new_state, reward, occupied = env[j].time_step(actions[j], transmit_or_wait_s[j], sum(transmit_or_wait_s), ack)  # CHANNEL
        rewards[j] = reward
        score[j].append(reward)
        env[j].new_state = new_state

    for j in range(number_of_agents):
        np.random.seed(j)
        #print('Agent ', j)
        #draw.render_Q_diffs(agent[j].Q[:, :, 0], agent[j].Q[:, :, 1], j,i,env[j].state,actions[j], rewards[j], env[j].new_state)
        if i == 10:
            epsilon[0] = 1
        if i > 20:
            epsilon[0] = 0
        actions[j], transmit_or_wait_s[j] = agent[j].step(env[j].state, rewards[j], actions[j], transmit_or_wait_s[j], env[j].new_state, epsilon[j])
        
        if re_explore:
            epsilon[j] = epsilon[j]*decay_rate
        if i > 1000000:
            if len(active) == 3:
                for c in active:
                    actions[c], transmit_or_wait_s[c] = 0, 0
            else:
                if actions[j] == 1:
                    active.append(j)
        #if np.mean(score[0][-1000:-1]) < r_max and epsilon[0] == 0:
        #    epsilon = np.ones(number_of_agents)*0.01
        #    re_explore = True
            #r_max = np.mean(score[0][-1000:-1])
        
    if state_transition_graph:
        # collect state transitions in T
        for j in range(number_of_agents):
            # decompose state
            current_energy, slient_time, idle_time= env[j].state
            # decompose new state
            next_energy, next_silence, next_idle_time = env[j].new_state
            # print(current_energy, slient_time,'->',next_energy, next_silence , '~~~', current_energy*(BATTERY_SIZE-1)+slient_time, next_energy*(BATTERY_SIZE-1)+next_silence)
            T[j][current_energy * (BATTERY_SIZE) + slient_time * (MAX_SILENT_TIME) + idle_time, next_energy * (BATTERY_SIZE) + next_silence * (MAX_SILENT_TIME) + next_idle_time] += 1

    #for a in range(number_of_agents):
    #    new_state, reward, occupied = env[a].time_step(actions[a],transmit_or_wait_s[a], sum(transmit_or_wait_s), ack)  # CHANNEL
    #    env[a].new_state = new_state
    #    actions[a] ,transmit_or_wait_s[a] = agent[a].choose_action(env[a].new_state, 0)#step(env[a].state, reward, actions[a],transmit_or_wait_s[a], env[a].new_state, epsilon[a])
print('collisions', collisions)
print('wasted', wasted)
for a in range(number_of_agents):
    print('agent{d}'.format(d=a), agent_clean[a] , 'rate:  ', env[a].discharge_rate)
   
for a in range(number_of_agents):
    for i in range(int(len(score[a])/1000)):
        avg_rwrd[a].append(np.mean(score[a][1000*i:1000*(i+1)]))
    plt.plot(range(len(avg_rwrd[a])), avg_rwrd[a])
plt.legend(range(number_of_agents))
plt.show()
'''
#print(data)
if evaluation_slots:
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
if state_transition_graph:
    draw.draw_state_transition_graph(T,agent)