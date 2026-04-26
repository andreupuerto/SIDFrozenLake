import numpy as np
import config

class ReinforceAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = config.GAMMA
        self.learning_rate = config.REINFORCE_LR
        self.lr_decay = config.LR_DECAY
        self.policy_table = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        self.env.action_space.seed(config.SEED)
        np.random.seed(config.SEED)
        self.env.observation_space.seed(config.SEED)

    def select_action(self, state, training=True):
        action_probabilities = self.policy_table[state]
        if training:
            return np.random.choice(np.arange(self.env.action_space.n), p=action_probabilities)
        else:
            return np.argmax(action_probabilities)

    def update_policy(self, episode):
        states, actions, rewards = episode
        discounted_rewards = np.zeros_like(rewards, dtype=float)
        running_add = 0
        
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)
        loss = -np.sum(np.log(self.policy_table[states, actions] + 1e-10) * discounted_rewards) / len(states)
        policy_logits = np.log(self.policy_table + 1e-10)
        for t in range(len(states)):
            G_t = discounted_rewards[t]
            action_probs = np.exp(policy_logits[states[t]])
            action_probs /= np.sum(action_probs)
            policy_gradient = G_t * (1 - action_probs[actions[t]])
            policy_logits[states[t], actions[t]] += self.learning_rate * policy_gradient
        exp_logits = np.exp(policy_logits)
        self.policy_table = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return loss

    def train(self):
        history = []
        for i in range(config.NUM_EPISODES):
            state, _ = self.env.reset()
            episode_data = []
            done = False
            truncated = False
            total_reward = 0
            step = 0
            
            while not (done or truncated) and step < config.T_MAX:
                action = self.select_action(state, training=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                episode_data.append((state, action, reward))
                state = next_state
                total_reward += reward
                step += 1
            
            if episode_data:
                loss = self.update_policy(zip(*episode_data))
            
            self.learning_rate *= self.lr_decay
            
            history.append(total_reward)
            
            if (i + 1) % 100 == 0:
                print(f"Episodio {i+1}/{config.NUM_EPISODES} - Recompensa media: {np.mean(history[-100:]):.4f}")
                
        return len(history)

    def get_policy(self):
        return np.argmax(self.policy_table, axis=1)