import numpy as np
import config

class ReinforceAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.learning_rate = config.REINFORCE_LR
        self.lr_decay = config.LR_DECAY
        self.policy_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        self.policy_logits = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def _softmax(self, logits):
        shifted = logits - np.max(logits) 
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state, training=True):
        probs = self._softmax(self.policy_logits[state])
        if training:
            return np.random.choice(self.env.action_space.n, p=probs)
        else:
            return np.argmax(probs)

    def get_policy(self):
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for state in range(self.env.observation_space.n):
            policy[state] = np.argmax(self._softmax(self.policy_logits[state]))
        return policy

    def _compute_returns(self, rewards):
        returns = np.zeros(len(rewards))
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            returns[t] = running_add

        return returns

    def update_policy(self, states, actions, rewards):
        returns = self._compute_returns(rewards)

        for t in range(len(states)):
            state_t = states[t]
            action_t = actions[t]
            G_t = returns[t]

            probs = self._softmax(self.policy_logits[state_t])
            grad = -probs.copy()
            grad[action_t] += 1

            self.policy_logits[state_t] += self.learning_rate * G_t * grad

    def train(self):
        history = []

        for i in range(config.NUM_EPISODES):
            state, _ = self.env.reset()
            states, actions, rewards = [], [], []
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < config.T_MAX:
                action = self.select_action(state, training=True)
                next_state, reward, done, truncated, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                step += 1

            if states:
                self.update_policy(states, actions, rewards)

            self.learning_rate *= self.lr_decay
            total_reward = sum(rewards)
            history.append(total_reward)

            if (i + 1) % 100 == 0:
                mean_reward = np.mean(history[-100:])
                print(f"Episodio {i+1}/{config.NUM_EPISODES} - Recompensa media: {mean_reward:.4f}")

        return history
