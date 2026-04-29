import numpy as np
import config

class ValueIterationAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.V = np.zeros(self.env.observation_space.n)

    def calc_action_value(self, state, action):
        action_value = sum([
            prob * (reward + self.gamma * self.V[next_state])
            for prob, next_state, reward, _ 
            in self.env.unwrapped.P[state][action]
        ])
        return action_value

    def select_action(self, state, training=False):
        best_action = None
        best_value = -float('inf')
        
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def train(self):
        iterations = 0
        while True:
            max_diff = 0
            for state in range(self.env.observation_space.n):
                v_old = self.V[state]
                
                state_values = [
                    self.calc_action_value(state, action) 
                    for action in range(self.env.action_space.n)
                ]
                
                self.V[state] = max(state_values)
                
                max_diff = max(max_diff, abs(v_old - self.V[state]))
            
            iterations += 1
            
            if max_diff < config.THETA_CONVERGENCE:
                break
                
        return iterations

    def get_policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for s in range(self.env.observation_space.n):
            q_values = [self.calc_action_value(s, a) for a in range(self.env.action_space.n)]
            policy[s] = np.argmax(q_values)
        return policy