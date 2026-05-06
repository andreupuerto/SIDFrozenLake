# main.py
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper

import config
from value_iteration import ValueIterationAgent
from reinforce import ReinforceAgent
from qlearning import QLearningAgent
from model_based import ModelBasedAgent


# ── Entorno ───────────────────────────────────────────────────────────────────

class CustomFrozenLakeWrapper(Wrapper):
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if not config.DEFAULT_REWARD:
            if terminated and reward < 1:
                reward = config.HOLE_PENALTY
            elif terminated and reward >= 1:
                reward = config.GOAL_REWARD
            elif not terminated:
                reward = config.STEP_PENALTY
        return state, reward, terminated, truncated, info


def create_env(map_name=None, success_rate=None, render_mode=None):
    m_name = map_name if map_name is not None else config.MAP_NAME
    sr = success_rate if success_rate is not None else config.SUCCESS_RATE
    env = gym.make(
        "FrozenLake-v1",
        map_name=m_name,
        is_slippery=True,
        success_rate=sr,
        render_mode=render_mode,
    )
    return CustomFrozenLakeWrapper(env)


# ── Evaluación y plots ────────────────────────────────────────────────────────

def evaluate_agent(agent, env, num_episodes=None):
    episodes = num_episodes if num_episodes is not None else config.NUM_EPISODES_TEST
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = truncated = False
        ep_reward = 0
        steps = 0
        while not (done or truncated) and steps < config.T_MAX:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
        if done and not truncated and ep_reward > 0:
            successes += 1
        total_rewards.append(ep_reward)

    return (successes / episodes) * 100, np.mean(total_rewards)


def plot_learning(history, name, map_name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, color="blue", label="Recompensa por episodio")
    if len(history) >= 100:
        smooth = np.convolve(history, np.ones(100) / 100, mode="valid")
        plt.plot(smooth, color="red", label="Media móvil (100 ep.)")
    title = f"Progreso - {name}" + (f" ({map_name})" if map_name else "")
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── Construcción de agentes ───────────────────────────────────────────────────

def build_agent(name, env):
    if name == "value_iteration":
        return ValueIterationAgent(env)
    elif name == "reinforce":
        return ReinforceAgent(env)
    elif name == "qlearning":
        return QLearningAgent(env)
    elif name == "model_based":
        return ModelBasedAgent(env)
    else:
        raise ValueError(f"Agente desconocido: {name}")


# ── Ejecución interactiva ─────────────────────────────────────────────────────

def interactive_config():
    print("\n--- CONFIGURACIÓN ---")
    mapa = input(f"Mapa (4x4/8x8) [actual: {config.MAP_NAME}]: ").strip()
    if mapa in ["4x4", "8x8"]:
        config.MAP_NAME = mapa
    sr = input(f"Success rate 0-1 [actual: {config.SUCCESS_RATE}]: ").strip()
    try:
        config.SUCCESS_RATE = float(sr)
    except ValueError:
        pass
    ep = input(f"Nº episodios [actual: {config.NUM_EPISODES}]: ").strip()
    if ep.isdigit():
        config.NUM_EPISODES = int(ep)


if __name__ == "__main__":
    interactive_config()

    print("\n" + "=" * 40)
    print(f" Mapa: {config.MAP_NAME} | SR: {config.SUCCESS_RATE}")
    print("=" * 40)
    print("1. Value Iteration")
    print("2. REINFORCE")
    print("3. Q-Learning")
    print("4. Model Based")

    opciones = {"1": "value_iteration", "2": "reinforce",
                "3": "qlearning", "4": "model_based"}
    opcion = input("\nSelecciona algoritmo: ").strip()

    if opcion not in opciones:
        print("Opción no válida.")
        exit(1)

    env = create_env()
    agent = build_agent(opciones[opcion], env)

    print(f"\nEntrenando {agent.__class__.__name__}...")
    start = time.time()
    resultado = agent.train()
    elapsed = time.time() - start
    print(f"Completado en {elapsed:.2f}s")

    if isinstance(resultado, list):
        plot_learning(resultado, agent.__class__.__name__, config.MAP_NAME)
    else:
        print(f"Convergencia en {resultado} iteraciones.")

    pct, avg_r = evaluate_agent(agent, env)
    print(f"\nÉxito: {pct:.2f}% | Recompensa media: {avg_r:.4f}")
    env.close()
