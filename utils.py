# utils.py
import numpy as np
import matplotlib.pyplot as plt
import config


def evaluate_agent(agent, env, num_episodes=None):
    """
    Evalúa un agente entrenado durante num_episodes episodios.
    Devuelve el porcentaje de episodios exitosos y la recompensa media.
    """
    episodes = num_episodes if num_episodes is not None else config.NUM_EPISODES_TEST
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0

        while not (done or truncated) and steps < config.T_MAX:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1

        # Éxito = episodio terminado sin truncar (llegó a la meta o cayó en hoyo)
        # como HOLE_PENALTY < 0 y GOAL_REWARD > 0, ep_reward > 0 implica meta
        if done and not truncated and ep_reward > 0:
            successes += 1

        total_rewards.append(ep_reward)

    success_rate = (successes / episodes) * 100
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward


def plot_learning(history, name, map_name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, color="blue", label="Recompensa por episodio")
    if len(history) >= 100:
        smooth = np.convolve(history, np.ones(100) / 100, mode="valid")
        plt.plot(smooth, color="red", label="Media móvil (100 ep.)")
    title = f"Progreso de Entrenamiento - {name}"
    if map_name:
        title += f" ({map_name})"
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
