import numpy as np
import collections
import config


class ModelBasedAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.state, _ = self.env.reset()

    def _play_random_steps(self, count):
        """Explora aleatoriamente para estimar las funciones de transición y recompensa."""
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done or truncated:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        if total == 0:
            return 0.0
        action_value = 0.0
        for next_state, count in target_counts.items():
            r = self.rewards[(state, action, next_state)]
            prob = count / total
            action_value += prob * (r + self.gamma * self.V[next_state])
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

    def _value_iteration_step(self):
        """Una iteración de value iteration sobre el modelo estimado."""
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return max_diff

    def train(self):
        """
        Entrena el agente alternando exploración aleatoria e iteración de valor
        hasta convergencia por THETA_CONVERGENCE.
        A diferencia de Value Iteration, el MDP se estima desde experiencia,
        no se lee directamente.
        """
        # Exploración inicial para tener datos suficientes antes de iterar
        self._play_random_steps(1000)

        history = []
        while True:
            # Exploración adicional por iteración
            self._play_random_steps(100)
            max_diff = self._value_iteration_step()
            history.append(max_diff)

            if max_diff < config.THETA_CONVERGENCE:
                break

        return history

    def get_policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for s in range(self.env.observation_space.n):
            q_values = [self.calc_action_value(s, a)
                        for a in range(self.env.action_space.n)]
            policy[s] = np.argmax(q_values)
        return policy
