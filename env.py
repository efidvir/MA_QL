import gym
from gym import spaces
import numpy as np

class transmit_env(gym.Env):
    def __init__(self, battery_size, max_silence_time, time_threshold, minimal_charge, discharge_rate, charge_rate,
                 data_size,action_space_size):
        super(transmit_env, self).__init__()

        # set env parameters
        self.battery_size = battery_size
        self.max_silence_time = max_silence_time
        self.time_threshold = time_threshold
        self.minimal_charge = minimal_charge
        self.discharge_rate = discharge_rate
        self.charge_rate = charge_rate
        self.data_size = data_size

        # reward functions
        '''
        self.r_1 = np.append(np.zeros(self.time_threshold-1),-1*np.ones(self.max_silence_time  - self.time_threshold))
        self.r_1 = np.append(np.zeros(self.time_threshold - 1),
                             -1 * np.linspace(0, 2 * (self.max_silence_time - self.time_threshold),
                                              self.max_silence_time + 1 - self.time_threshold + 1))
        '''

        # action space
        self.action_space_size = action_space_size
        self.action_space = spaces.Discrete(action_space_size)

        # state space
        self.state_space = spaces.Tuple((spaces.Discrete(self.max_silence_time), spaces.Discrete(self.battery_size)))
        self.initial_energy = self.battery_size
        self.initial_silence = 0
        self.initial_state = [self.initial_energy - 1, self.initial_silence]
        self.state = self.initial_state
        self.new_state = self.initial_state

        # Screen size of data
        #self.screen = pygame.display.set_mode((data_size * 100, 100))

    def time_step(self, action,transmit_or_wait, channel, ack, ):
        # take action accoring to policy (epsilon-greedy) and get reward and next state
        #######################################################
        current_energy, silent_time = self.state
        reward = 0
        if ack:
            reward += 1
        if transmit_or_wait == 1:  # agent choose to transmit and discharge
            if current_energy < self.minimal_charge:
                raise ValueError('No charge left, can not transmit')
            else:
                current_energy -= self.discharge_rate

            if channel > 1:  # Someone else transmited along with agent - collision
                silent_time += 1
                occupied = 1
            else:  # Agent made a sucsessful report!
                  # Gateway responded!
                    silent_time = 0
                    #reward += 1
                #else:
                #    raise ValueError('Gateway not responding')

        else:  # agent choose to wait and charge
            if current_energy < self.battery_size - 1:  # capp battery
                current_energy += self.charge_rate
            if ack:  # someone made a sucsessful report!
                silent_time = 0
            else:
                silent_time += 1

        if silent_time > self.max_silence_time - 1:  # capp time
            silent_time = self.max_silence_time - 1

        if channel == 0:
            occupied = 0
            reward -= 1
        elif channel == 1 and action:
            occupied = 0
        else:
            occupied = 1

        #reward += self.get_reward(current_energy, silent_time)

        # compose new state
        new_state = [current_energy, silent_time]

        return new_state, reward, occupied

    def reset():  #EPISODIC NOT IMPLEMENTED - PURE ONLINE
        return

    def get_reward(self, energy, silenct_time):
        reward = self.r_1[silenct_time]  # + self.r_3[energy]
        return reward

    def get_channel_occupency(self, loss):
        event = np.random.uniform(0, 1, 1)
        return event < loss