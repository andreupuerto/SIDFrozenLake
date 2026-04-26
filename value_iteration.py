import numpy as np
import config

class ValueIterationAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.V = np.zeros(self.env.observation_space.n)

    def calc_action_value(self, state, action):
        action_value = 0
        for prob, next_state, reward, terminated in self.env.unwrapped.P[state][action]:
            action_value += prob * (reward + self.gamma * self.V[next_state])
        return action_value

    def select_action(self, state, training=False):
        q_values = [self.calc_action_value(state, a) for a in range(self.env.action_space.n)]
        return np.argmax(q_values)

    def train(self):
        iterations = 0
        while True:
            max_diff = 0
            for s in range(self.env.observation_space.n):
                v_old = self.V[s]
                self.V[s] = max([self.calc_action_value(s, a) for a in range(self.env.action_space.n)])
                max_diff = max(max_diff, abs(v_old - self.V[s]))
                iterations += 1
            if max_diff < config.THETA_CONVERGENCE:
                break
        return iterations