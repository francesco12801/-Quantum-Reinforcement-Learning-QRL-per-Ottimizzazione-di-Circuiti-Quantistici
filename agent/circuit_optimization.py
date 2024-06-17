import numpy as np

from set_quantum import * 



class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate, discount_factor, epsilon):
        self.q_table = np.zeros((state_space, action_space))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Explore 
        else:
            return np.argmax(self.q_table[state])  

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)
