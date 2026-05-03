# utils.py - Funciones para graficas y metricas
import numpy as np
import matplotlib.pyplot as plt
import config


def evaluate_agent(agent, env, num_episodes=None):
    # evalua el agente sin que aprenda (solo test)
    episodes = num_episodes if num_episodes is not None else config.NUM_EPISODES_TEST
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0

        # bucle del episodio
        while not (done or truncated) and steps < config.T_MAX:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1

        # si llega al regalo cuenta como exito
        if done and not truncated and ep_reward > 0:
            successes += 1

        total_rewards.append(ep_reward)

    success_rate = (successes / episodes) * 100
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward


def plot_learning(history, name, map_name=None):
    # hace el plot de las recompensas por episodio
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, color="blue", label="Recompensa por episodio")
    
    # media movil para que se vea mas suave la curva
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
