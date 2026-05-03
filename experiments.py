# experiments.py — batería completa de experimentos
#
# Uso:
#   python experiments.py                  # corre todos los experimentos
#   python experiments.py --exp 1          # solo el experimento 1
#   python experiments.py --exp 3 --save   # exp 3 y guarda gráficas en results/
#

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config
from environment import create_env
from utils import evaluate_agent
from value_iteration import ValueIterationAgent
from reinforce import ReinforceAgent

# [PENDING] from qlearning import QLearningAgent
# [PENDING] from model_based import ModelBasedAgent

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")


def setup_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)

ACTIVE_AGENTS = [
    "Value Iteration",
    "REINFORCE",
    # "Q-Learning",    # [PENDING]
    # "Model Based",   # [PENDING]
]

# Solo estos agentes tienen num_episodes como hiperparámetro relevante
EPISODE_DEPENDENT = {"REINFORCE", "Q-Learning"}


def build_agent(name, env):
    if name == "Value Iteration":
        return ValueIterationAgent(env)
    elif name == "REINFORCE":
        return ReinforceAgent(env)
    # [PENDING] elif name == "Q-Learning":
    #     return QLearningAgent(env)
    # [PENDING] elif name == "Model Based":
    #     return ModelBasedAgent(env)
    else:
        raise ValueError(f"Agente desconocido: {name}")


def train_agent(agent):
    start = time.time()
    resultado = agent.train()
    elapsed = time.time() - start
    history = resultado if isinstance(resultado, list) else []
    return history, elapsed


def run_agents(agents_to_run, map_name, success_rate, gamma, num_episodes, save, label):
    config.GAMMA = gamma
    config.NUM_EPISODES = num_episodes

    env = create_env(map_name=map_name, success_rate=success_rate)
    results = {}
    histories = {}

    for name in agents_to_run:
        print(f"    [{name}] entrenando...", end=" ", flush=True)
        agent = build_agent(name, env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        print(f"éxito={pct:.1f}% | reward={avg_r:.4f} | tiempo={elapsed:.2f}s")

        results[name] = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories[name] = history

        if save:
            _append_csv(label, map_name, success_rate, gamma, num_episodes,
                        name, pct, avg_r, elapsed)

    env.close()

    if save:
        _plot_bars(results, f"% Éxito — {label}", f"{label}_success.png")
        _plot_times(results, f"Tiempo — {label}", f"{label}_time.png")
        _plot_curves(histories, f"Curvas — {label}", f"{label}_curves.png")

    return results, histories


def exp1_map_size(save=False):
    print("\n" + "=" * 55)
    print("EXPERIMENTO 1 — Tamaño del mapa")
    print("=" * 55)

    maps = ["4x4", "8x8"]
    sr = 0.8
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    all_results = {}
    for m in maps:
        print(f"\n  Mapa={m} | SR={sr} | γ={gamma} | ep={num_ep}")
        res, _ = run_agents(ACTIVE_AGENTS, m, sr, gamma, num_ep, save, f"exp1_map{m}")
        all_results[m] = res

    if save:
        _plot_map_comparison(all_results, maps,
                             "Exp1: % Éxito por tamaño de mapa", "exp1_comparison.png")
    return all_results


def exp2_success_rate(save=False):
    print("\n" + "=" * 55)
    print("EXPERIMENTO 2 — Efecto del success_rate")
    print("=" * 55)

    success_rates = [0.33, 0.5, 0.8, 1.0]
    map_name = "4x4"
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    all_results = {}
    for sr in success_rates:
        print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
        res, _ = run_agents(ACTIVE_AGENTS, map_name, sr, gamma, num_ep,
                            save, f"exp2_sr{sr}")
        all_results[sr] = res

    if save:
        _plot_line_comparison(all_results, success_rates, "Success Rate",
                              "Exp2: % Éxito vs Success Rate (4x4)",
                              "exp2_sr_comparison.png")
    return all_results


def exp3_combined(save=False):
    """Experimento 3: Combinado mapa x estocasticidad. Todos los algoritmos."""
    print("\n" + "=" * 55)
    print("EXPERIMENTO 3 — Combinado: mapa + estocasticidad")
    print("=" * 55)

    combos = [("4x4", 0.33), ("4x4", 0.8), ("8x8", 0.33), ("8x8", 0.8)]
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    all_results = {}
    for map_name, sr in combos:
        print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
        label = f"exp3_map{map_name}_sr{sr}"
        res, _ = run_agents(ACTIVE_AGENTS, map_name, sr, gamma, num_ep, save, label)
        all_results[(map_name, sr)] = res

    if save:
        _plot_combined_heatmap(all_results, combos,
                               "Exp3: % Éxito (mapa x estocasticidad)",
                               "exp3_combined.png")
    return all_results


def exp4_gamma(save=False):
    """Experimento 4: Efecto de gamma. Mapa 4x4. Todos los algoritmos."""
    print("\n" + "=" * 55)
    print("EXPERIMENTO 4 — Efecto de gamma")
    print("=" * 55)

    gammas = [0.90, 0.95, 0.99]
    map_name = "4x4"
    sr = 0.8
    num_ep = config.NUM_EPISODES

    all_results = {}
    for g in gammas:
        print(f"\n  Mapa={map_name} | SR={sr} | γ={g} | ep={num_ep}")
        res, _ = run_agents(ACTIVE_AGENTS, map_name, sr, g, num_ep,
                            save, f"exp4_g{g}")
        all_results[g] = res

    if save:
        _plot_line_comparison(all_results, gammas, "Gamma (γ)",
                              "Exp4: % Éxito vs Gamma",
                              "exp4_gamma_comparison.png")
    return all_results


def exp5_episodes(save=False):
    """
    Experimento 5: Efecto del número de episodios.
    Solo Q-Learning y REINFORCE (agentes episódicos).
    """
    print("\n" + "=" * 55)
    print("EXPERIMENTO 5 — Número de episodios (agentes episódicos)")
    print("=" * 55)

    episode_counts = [500, 1000, 2000, 5000]
    map_name = "4x4"
    sr = 0.8
    gamma = config.GAMMA

    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]
    if not episodic_active:
        print("  (ningún agente episódico activo — añade Q-Learning o REINFORCE)")
        return {}

    all_results = {}
    all_histories = {}
    for num_ep in episode_counts:
        print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
        res, hist = run_agents(episodic_active, map_name, sr, gamma, num_ep,
                               save, f"exp5_ep{num_ep}")
        all_results[num_ep] = res
        all_histories[num_ep] = hist

    if save:
        _plot_line_comparison(all_results, episode_counts, "Nº episodios",
                              "Exp5: % Éxito vs Nº episodios",
                              "exp5_episodes_comparison.png")
        # Curvas de aprendizaje para el mayor número de episodios
        _plot_curves(all_histories[episode_counts[-1]],
                     f"Exp5: Curvas con {episode_counts[-1]} episodios",
                     f"exp5_curves_{episode_counts[-1]}ep.png")

    return all_results


# ── Gráficas ──────────────────────────────────────────────────────────────────

COLORS = ["steelblue", "seagreen", "tomato", "mediumpurple", "darkorange"]


def _plot_bars(results_by_agent, title, filename):
    agents = list(results_by_agent.keys())
    values = [results_by_agent[a]["success"] for a in agents]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, values, color=COLORS[:len(agents)])
    ax.set_title(title)
    ax.set_ylabel("% Episodios exitosos")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_times(results_by_agent, title, filename):
    agents = list(results_by_agent.keys())
    values = [results_by_agent[a]["time"] for a in agents]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, values, color=COLORS[:len(agents)])
    ax.set_title(title)
    ax.set_ylabel("Tiempo (s)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}s", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_curves(histories_by_agent, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, history), color in zip(histories_by_agent.items(), COLORS):
        if not history:
            continue
        ax.plot(history, alpha=0.2, color=color)
        if len(history) >= 100:
            smooth = np.convolve(history, np.ones(100) / 100, mode="valid")
            ax.plot(smooth, label=name, color=color)
        else:
            ax.plot(history, label=name, color=color)
    ax.set_title(title)
    ax.set_xlabel("Episodios")
    ax.set_ylabel("Recompensa")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_map_comparison(all_results, maps, title, filename):
    agent_names = list(all_results[maps[0]].keys())
    x = np.arange(len(agent_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(maps):
        vals = [all_results[m][a]["success"] for a in agent_names]
        ax.bar(x + i * width, vals, width, label=f"Mapa {m}", color=COLORS[i])
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(agent_names)
    ax.set_ylabel("% Éxito")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_line_comparison(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, color in zip(agent_names, COLORS):
        vals = [all_results[x][agent]["success"] for x in x_values]
        ax.plot(x_values, vals, marker="o", label=agent, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% Éxito")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_combined_heatmap(all_results, combos, title, filename):
    agent_names = list(all_results[combos[0]].keys())
    combo_labels = [f"{m}\nSR={sr}" for m, sr in combos]

    fig, axes = plt.subplots(1, len(agent_names), figsize=(5 * len(agent_names), 5))
    if len(agent_names) == 1:
        axes = [axes]

    for ax, agent in zip(axes, agent_names):
        vals = [all_results[c][agent]["success"] for c in combos]
        bars = ax.bar(combo_labels, vals, color=COLORS[:len(combos)])
        ax.set_title(agent)
        ax.set_ylabel("% Éxito")
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


# ── CSV ───────────────────────────────────────────────────────────────────────

def _init_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp", "map_name", "success_rate", "gamma",
                         "num_episodes", "agent", "pct_success",
                         "avg_reward", "train_time_s"])


def _append_csv(exp, map_name, sr, gamma, num_ep, agent, pct, avg_r, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([exp, map_name, sr, gamma, num_ep, agent,
                         f"{pct:.2f}", f"{avg_r:.4f}", f"{elapsed:.2f}"])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Experimentos FrozenLake")
    parser.add_argument("--exp", type=int, default=0,
                        help="Experimento a ejecutar (1-5). 0 = todos.")
    parser.add_argument("--save", action="store_true",
                        help="Guarda gráficas y CSV en results/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_dirs()

    if args.save:
        _init_csv()

    runners = {
        1: exp1_map_size,
        2: exp2_success_rate,
        3: exp3_combined,
        4: exp4_gamma,
        5: exp5_episodes,
    }

    if args.exp == 0:
        for fn in runners.values():
            fn(save=args.save)
    elif args.exp in runners:
        runners[args.exp](save=args.save)
    else:
        print(f"Experimento {args.exp} no existe. Elige entre 1 y 5.")

    if args.save:
        print(f"\nResultados en '{RESULTS_DIR}/' | CSV en '{CSV_PATH}'")
