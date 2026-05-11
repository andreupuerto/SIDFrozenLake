# main.py
# Changes: rewrote environment creation to use Gymnasium's native
# success_rate/reward_schedule and custom-map TimeLimit support (Section 2.1),
# expanded evaluation metrics and termination accounting (Section 2.2), added VI
# oracle and coverage diagnostics (Sections 2.3-2.4), and removed interactive
# input so runs stay reproducible from code/CLI (Section 2.5).

import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import TimeLimit

import config
from model_based import ModelBasedAgent
from qlearning import QLearningAgent
from reinforce import ReinforceAgent
from value_iteration import ValueIterationAgent


# -- Entorno ------------------------------------------------------------------

def _parse_custom_map_size(map_name):
    try:
        left, right = str(map_name).lower().split("x")
        size = int(left)
        if size != int(right) or size <= 0:
            raise ValueError
        return size
    except (ValueError, AttributeError):
        raise ValueError(
            f"map_name debe ser '4x4', '8x8' o un mapa cuadrado tipo '10x10': {map_name}"
        ) from None


def create_env(map_name=None, success_rate=None, reward_schedule=None,
               render_mode=None, map_seed=None):
    """
    Crea FrozenLake-v1 con mapas nativos o custom NxN.

    reward_schedule se pasa directamente a gym.make(), evitando wrappers
    propios para reward shaping.
    """
    m_name = map_name if map_name is not None else config.MAP_NAME
    sr = success_rate if success_rate is not None else config.SUCCESS_RATE
    rs = reward_schedule if reward_schedule is not None else config.REWARD_SCHEDULE

    kwargs = {
        "is_slippery": config.SLIPPERY,
        "success_rate": sr,
        "reward_schedule": rs,
        "render_mode": render_mode,
    }

    if m_name in {"4x4", "8x8"}:
        kwargs["map_name"] = m_name
        return gym.make("FrozenLake-v1", **kwargs)

    size = _parse_custom_map_size(m_name)
    seed = map_seed if map_seed is not None else config.SEED
    kwargs["desc"] = generate_random_map(size=size, seed=seed)
    env = gym.make("FrozenLake-v1", **kwargs)
    return TimeLimit(env, max_episode_steps=config.get_t_max(m_name))


# -- Evaluacion y diagnosticos ------------------------------------------------

def evaluate_agent(agent, env, num_episodes=None):
    """
    Evalua un agente con politica determinista y devuelve las metricas del plan.
    """
    episodes = num_episodes if num_episodes is not None else config.NUM_EPISODES_TEST

    successes = 0
    holes = 0
    timeouts = 0
    total_rewards = []
    total_steps = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        last_reward = 0.0

        while not (done or truncated):
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            last_reward = reward
            steps += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)

        if truncated:
            timeouts += 1
        elif done and last_reward > 0:
            successes += 1
        elif done:
            holes += 1

    return {
        "eval_success_rate": (successes / episodes) * 100,
        "eval_mean_reward": float(np.mean(total_rewards)),
        "eval_mean_steps": float(np.mean(total_steps)),
        "eval_termination_goal_pct": (successes / episodes) * 100,
        "eval_termination_hole_pct": (holes / episodes) * 100,
        "eval_truncation_pct": (timeouts / episodes) * 100,
    }


def evaluate_agent_with_oracle(agent, env):
    """
    Compara la politica aprendida contra un oraculo de Value Iteration.
    """
    vi_oracle = ValueIterationAgent(env)
    vi_oracle.train()

    learned_policy = agent.get_policy()
    optimal_policy = vi_oracle.get_policy()
    agreement = np.mean(learned_policy == optimal_policy) * 100

    q_diff = None
    if hasattr(agent, "Q") and hasattr(vi_oracle, "V"):
        q_optimal = np.zeros((env.observation_space.n, env.action_space.n))
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                q_optimal[state, action] = vi_oracle.calc_action_value(state, action)
        q_diff = float(np.max(np.abs(agent.Q - q_optimal)))

    return {
        "eval_policy_agreement_vi": float(agreement),
        "final_q_diff_l_inf": q_diff,
    }


def compute_state_action_coverage(agent, env):
    """
    Calcula el porcentaje de pares (s,a) visitados durante el entrenamiento.
    """
    n_total = env.observation_space.n * env.action_space.n

    if hasattr(agent, "transits"):
        n_visited = len(agent.transits)
    elif hasattr(agent, "Q"):
        n_visited = int(np.sum(agent.Q != 0))
    else:
        return None

    return (n_visited / n_total) * 100


def plot_learning(history, name, map_name=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, color="blue", label="Recompensa por episodio")
    if len(history) >= 100:
        smooth = np.convolve(history, np.ones(100) / 100, mode="valid")
        plt.plot(smooth, color="red", label="Media movil (100 ep.)")
    title = f"Progreso - {name}" + (f" ({map_name})" if map_name else "")
    plt.title(title)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -- Construccion de agentes --------------------------------------------------

def build_agent(name, env):
    if name == "value_iteration":
        return ValueIterationAgent(env)
    if name == "reinforce":
        return ReinforceAgent(env)
    if name == "qlearning":
        return QLearningAgent(env)
    if name == "model_based":
        return ModelBasedAgent(env)
    raise ValueError(f"Agente desconocido: {name}")


if __name__ == "__main__":
    env = create_env()
    agent = build_agent("qlearning", env)

    print("\n" + "=" * 40)
    print(f"Mapa: {config.MAP_NAME} | SR: {config.SUCCESS_RATE:.3f}")
    print("=" * 40)
    print(f"Entrenando {agent.__class__.__name__}...")

    start = time.time()
    result = agent.train()
    elapsed = time.time() - start
    print(f"Completado en {elapsed:.2f}s")

    if isinstance(result, list):
        plot_learning(result, agent.__class__.__name__, config.MAP_NAME)
    else:
        print(f"Convergencia en {result} iteraciones.")

    metrics = evaluate_agent(agent, env)
    print(
        f"Exito: {metrics['eval_success_rate']:.2f}% | "
        f"Recompensa media: {metrics['eval_mean_reward']:.4f}"
    )
    env.close()
