import numpy as np
import random
import config


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.learning_rate = config.LEARNING_RATE
        self.lr_decay = config.LR_DECAY
        self.epsilon = config.EPSILON
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_MIN
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.Q[state])

    def _update_Q(self, state, action, reward, next_state, done):
        best_next = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next] * (not done)
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error

    def _learn_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = truncated = False
        steps = 0

        while not (done or truncated) and steps < config.T_MAX:
            action = self.select_action(state, training=True)
            next_state, reward, done, truncated, _ = self.env.step(action)
            self._update_Q(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        return total_reward

    def train(self):
        """
        Entrena el agente durante NUM_EPISODES episodios.
        Aplica decay de epsilon y learning_rate tras cada episodio
        para pasar progresivamente de exploración a explotación.
        """
        history = []

        for i in range(config.NUM_EPISODES):
            total_reward = self._learn_episode()
            history.append(total_reward)

            # Decay de epsilon y learning rate tras cada episodio
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)
            self.learning_rate *= self.lr_decay

            if (i + 1) % 100 == 0:
                mean_reward = np.mean(history[-100:])
                print(f"Episodio {i+1}/{config.NUM_EPISODES} "
                      f"| Recompensa media: {mean_reward:.4f} "
                      f"| ε={self.epsilon:.4f} "
                      f"| α={self.learning_rate:.4f}")

        return history

    def get_policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for s in range(self.env.observation_space.n):
            policy[s] = np.argmax(self.Q[s])
        return policy
