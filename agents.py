import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Layer, Dense, Softmax
from tensorflow.python.keras.optimizer_v2 import adam
#import tensorflow_probability as tfp
import os, sys, time


class Q_transmit_agent():
    def __init__(self, alpha, gamma, battery_size, max_silence_time, data_size, number_of_actions,MINIMAL_CHARGE,RAND):
        self.alpha = alpha
        self.beta = alpha#/10
        self.gamma = gamma
        self.data_size = data_size
        self.idle_time = 2
        self.number_of_actions = number_of_actions
        self.Q = np.zeros(shape=(battery_size, max_silence_time, self.idle_time, self.number_of_actions))
        self.state_visits = np.zeros(shape=(battery_size, max_silence_time, self.idle_time))
        self.error = np.zeros(shape=(battery_size, max_silence_time,self.idle_time, self.number_of_actions))
        self.MINIMAL_CHARGE = MINIMAL_CHARGE
        self.seeder = RAND
        self.priority = 0


    def choose_action(self, state, epsilon):
        # decompose state
        current_energy, slient_time, priority = state

        # Explore ?
        if np.random.default_rng().uniform(size=1)[0] < epsilon:
            np.random.seed(self.seeder[0]+int(time.time_ns()%1000000))
            action = np.random.default_rng().choice([0,1],1,p=[0.5,0.5])
            #action = np.random.randint(self.number_of_actions)

            #print('random action',np.random.uniform(size=1)[0] , epsilon)

        # Exploite - Choose the current best action
        else:
            action = np.argmax(self.Q[current_energy, slient_time, priority])  # Take the action that has the highest predicted Q value (0, 1)

        # Dont have energy for transmision#################################################
        if current_energy < self.MINIMAL_CHARGE:
            action = 0

        transmit_prob = action / (self.number_of_actions - 1)
        transmit_or_wait = action #np.random.choice([1, 0], p=(transmit_prob, 1 - transmit_prob))
        return action, transmit_or_wait

    def Q_learn(self, state, reward, action, new_state): ##########RENAME priority
        # decompose state
        current_energy, slient_time, priority = state
        # q_index = [current_energy,slient_time, action]
        #self.state_visits[current_energy, slient_time, priority] += 1
        delta = reward - self.Q[current_energy, slient_time, priority, action]

        # decompose new state
        next_energy, next_silence, next_priority = new_state
        # next_best_q_value_index = np.argmax(self.Q[next_energy, next_silence,:])
        # new_Q = reward + self.gamma*self.Q[next_energy, next_silence, next_best_q_value_index]
        # error = new_Q - self.Q[q_index]
        # self.Q[q_index] += self.alpha * error #################swap to alpha table
        self.error[current_energy, slient_time, priority,  action] = reward + self.gamma * (
            np.max(self.Q[next_energy, next_silence,next_priority,:])) - self.Q[current_energy, slient_time, priority,  np.argmax(
            self.Q[current_energy, slient_time, priority, action])]

        delta = reward + self.gamma * (np.max(self.Q[next_energy, next_silence, next_priority, :])) - self.Q[current_energy, slient_time, priority, action]
        if delta >=0:
            self.Q[current_energy, slient_time,priority, action] = self.Q[current_energy, slient_time, priority, action] + self.alpha * delta
        else:
            self.Q[current_energy, slient_time, priority, action] = self.Q[current_energy, slient_time, priority, action] + self.beta * delta
        #self.Q[current_energy, slient_time, action] = self.Q[current_energy, slient_time, action] + self.alpha * (
        #            reward + self.gamma * (np.max(self.Q[next_energy, next_silence, :])) - self.Q[
        #        current_energy, slient_time, action])
        return

    def step(self, state, reward, action,transmit_or_wait, new_state, epsilon):
        self.Q_learn(state, reward, action, new_state)
        new_action , new_transmit_or_wait = self.choose_action(new_state, epsilon)
        return new_action , new_transmit_or_wait

    def get_policy(self):
        policy = np.zeros(shape=(self.Q.shape[0], self.Q.shape[1]))
        for energy in range(self.Q.shape[0]):
            for time in range(self.Q.shape[1]):
                policy[energy, time] = np.argmax(self.Q[energy, time, :])
        return policy

    def get_state_value(self,policy):
        state_value = np.zeros(shape=(self.Q.shape[0], self.Q.shape[1]))
        for energy in range(self.Q.shape[0]):
            for time in range(self.Q.shape[1]):
                state_value[energy, time] = self.Q[energy, time, int(policy[energy, time])]
        return state_value

class ActorCriticNetwork(keras.Model):
    def __init__(self, number_of_actions, fc1_dims=128, fc2_dims =64, name='actor_critic', chkp_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.number_of_actions = number_of_actions
        self.model_name = name
        self.checkpoint_dir = chkp_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        self.fc1 = Dense(self.fc1_dims, activation = 'relu')
        self.fc2 = Dense(self.fc2_dims, activation = 'relu')
        self.v = Dense(1, activation = None)
        self.pi = Dense(self.number_of_actions, activation = 'softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        v = self.v(value)
        pi = self.pi(value)
        return v, pi

class AC_Agent():
    def __init__(self, alpha=0.0003, gamma=0.99, battery_size = 10, max_silence_time = 10, data_size = 1000, number_of_actions=2 ,MINIMAL_CHARGE = 0 ):
        self.gamma = gamma
        self.n_actions = number_of_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_critic = ActorCriticNetwork(number_of_actions=number_of_actions)
        self.actor_critic.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                  optimizer=adam.Adam(learning_rate=alpha))
        self.alpha = alpha

        self.data_size = data_size

        self.number_of_actions = number_of_actions
        self.Q = np.zeros(shape=(battery_size, max_silence_time, self.number_of_actions))
        self.state_visits = np.zeros(shape=(battery_size, max_silence_time))
        self.error = np.zeros(shape=(battery_size, max_silence_time, self.number_of_actions))
        self.MINIMAL_CHARGE = MINIMAL_CHARGE

    def choose_action(self, observation, epsilon):
        state = tf.convert_to_tensor([observation])
        energy, silent_time = observation
        self.state_visits[energy, silent_time] += 1
        _, probs = self.actor_critic(state)
        #print(probs)
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        if energy < self.MINIMAL_CHARGE:
            action = tf.convert_to_tensor([0], dtype=tf.float32)
        #log_prob = action_probabilities.log_prob(action)
        if action.numpy()[0]/10 > np.random.uniform(0, 1):
            transmit_or_wait = 1
        else:
            transmit_or_wait = 0
        self.action = action.numpy()[0]

        return action.numpy()[0], transmit_or_wait

    def step(self, state, reward, action,transmit_or_wait, new_state, epsilon):
        self.learn(state, reward, action, new_state)
        action , transmit_or_wait = self.choose_action(new_state, epsilon)
        return action , transmit_or_wait

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, reward, action, state_):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            delta = reward + self.gamma * state_value_ - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss
        #print('LOSSSSSSSSSSSSSSSSSSSSS:', total_loss)
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))

    def get_policy(self): # todo: change to non tabular generation
        pol = np.zeros(shape=(self.Q.shape[0], self.Q.shape[1]))
        for energy in range(self.Q.shape[0]):
            for time in range(self.Q.shape[1]):
                state = [energy, time]
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                state_value, policy = self.actor_critic.call(state)
                pol[energy, time] = np.argmax(policy.numpy()[0])
        return pol

    def get_state_value(self,policy): # todo: change to non tabular generation
        state_value = np.zeros(shape=(self.Q.shape[0], self.Q.shape[1]))
        for energy in range(self.Q.shape[0]):
            for time in range(self.Q.shape[1]):
                state = [energy, time]
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                v, policy = self.actor_critic.call(state)
                state_value[energy, time] = np.argmax(v.numpy()[0])
        return state_value

class simple_AC_Agent():
    def __init__(self, alpha=0.0003, gamma=0.99, battery_size = 10, max_silence_time = 10, data_size = 1000, number_of_actions=2 ,MINIMAL_CHARGE = 0 ):
        self.gamma = gamma
        self.n_actions = number_of_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        #self.actor_critic = ActorCriticNetwork(number_of_actions=number_of_actions)
        #self.actor_critic.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                  #optimizer=adam.Adam(learning_rate=alpha))
        self.alpha = alpha
        self.weights = weights
        self.value_weights = value_weights
        self.tau = tau

        self.data_size = data_size

        self.number_of_actions = number_of_actions
        self.Q = np.zeros(shape=(battery_size, max_silence_time, self.number_of_actions))
        self.state_visits = np.zeros(shape=(battery_size, max_silence_time))
        self.error = np.zeros(shape=(battery_size, max_silence_time, self.number_of_actions))
        self.MINIMAL_CHARGE = MINIMAL_CHARGE

    def __init__(self, number_of_actions, gamma, weights, value_weights, tau, learning_rate):
        self.actions = np.arange(number_of_actions)
        self.discount = gamma
        self.weights = weights
        self.value_weights = value_weights
        self.tau = tau
        self.learning_rate = learning_rate
        self.num_iters = 0

    def feature_extractor(self, state, action=None):
        return [((state, action), 1)]

    def getV(self, state):
        score = 0
        for f, v in self.feature_extractor(state):
            score += self.value_weights[f] * v
        return score

    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def get_action(self, state):
        """
        Softmax action selection.
        """
        self.num_iters += 1
        q_values = np.array([self.getQ(state, action) for action in self.actions])
        q_values = q_values - max(q_values)
        exp_q_values = np.exp(q_values / (self.tau + 1e-2))
        # should really be returning probs[action_idx]
        # sum_exp_q_values = np.sum(exp_q_values)
        # probs = exp_q_values / sum_exp_q_values
        weights = dict()
        for idx, val in enumerate(exp_q_values):
            weights[idx] = val
        action_idx = utils.weightedRandomChoice(weights)
        action = self.actions[action_idx]
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        """
        Update both actor and critic weights.
        """
        # prediction = V(s)
        prediction = self.getV(state)
        target = reward
        new_action = None

        if new_state != None:
            new_action = self.get_action(new_state)
            # target = r + yV(s')
            target += self.discount * self.getV(new_state)

        # advantage actor critic because we use the td error
        # as an unbiased sample of the advantage function
        update = self.learning_rate * (target - prediction)
        for f, v in self.feature_extractor(state):
            # update critic weights
            self.value_weights[f] = self.value_weights[f] + 2 * update

        for f, v in self.feature_extractor(state, action):
            # update actor weights
            # this update should actually be:
            # self.weights[f] += update * (v - prob(v))
            # since (v - prob(v)) is, in this case, equal to
            # the gradient of the log of the policy
            # however, that seems to work way worse than simply
            # multiplying by v (i.e., 1) instead, though it's likely
            # this version loses convergence gaurantees
            # and / or would work poorly with a neural net
            self.weights[f] = self.weights[f] + update * v

        return new_action