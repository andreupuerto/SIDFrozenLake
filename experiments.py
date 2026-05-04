# experiments.py — batería completa de experimentos
#
# Uso:
#   python experiments.py --exp calibration_gamma     # calibración gamma
#   python experiments.py --exp calibration_episodes  # calibración episodios
#   python experiments.py --exp main                  # experimento principal
#   python experiments.py --exp all                   # todos
#   python experiments.py --exp all --save            # todos + guarda resultados

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

# ── Directorios ───────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")

COLORS = {
    "Value Iteration": "steelblue",
    "REINFORCE":       "tomato",
    "Q-Learning":      "seagreen",
    "Model Based":     "mediumpurple",
}

# ── Agentes activos ───────────────────────────────────────────────────────────
# Descomenta cuando tus compañeros entreguen sus ficheros
ACTIVE_AGENTS = [
    "Value Iteration",
    "REINFORCE",
    # "Q-Learning",  # [PENDING]
    # "Model Based", # [PENDING]
]

# Solo estos tienen num_episodes como hiperparámetro relevante
EPISODE_DEPENDENT = {"REINFORCE", "Q-Learning"}


def setup_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Construcción de agentes ───────────────────────────────────────────────────

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


# ── Entrenamiento y evaluación ────────────────────────────────────────────────

def train_agent(agent):
    start = time.time()
    resultado = agent.train()
    elapsed = time.time() - start
    history = resultado if isinstance(resultado, list) else []
    return history, elapsed


def run_agents(agents_to_run, map_name, success_rate, gamma, num_episodes, save, label):
    """Entrena y evalúa cada agente con la configuración dada."""
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
            _append_csv(label, map_name, success_rate, gamma,
                        num_episodes, name, pct, avg_r, elapsed)

    env.close()
    return results, histories


# ── EXPERIMENTOS DE CALIBRACIÓN ───────────────────────────────────────────────

def calibration_gamma(save=False):
    """
    Exp A — Calibración de gamma.
    Fija: 4x4, SR=0.8, ep=config.NUM_EPISODES
    Varía: gamma = 0.90, 0.95, 0.99
    Objetivo: justificar el valor de gamma usado en el experimento principal.
    Hipótesis: gamma alto funciona mejor porque la meta está lejos y
               las rutas óptimas son largas.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN A — Efecto de gamma")
    print("=" * 55)

    gammas = [0.90, 0.95, 0.99]
    map_name = "4x4"
    sr = 0.8
    num_ep = config.NUM_EPISODES

    all_results = {}
    for g in gammas:
        print(f"\n  γ={g} | Mapa={map_name} | SR={sr} | ep={num_ep}")
        res, _ = run_agents(ACTIVE_AGENTS, map_name, sr, g, num_ep,
                            save, f"calA_g{g}")
        all_results[g] = res

    if save:
        _plot_line(all_results, gammas, "Gamma (γ)",
                   "Calibración A: % Éxito vs Gamma",
                   "calA_gamma.png")
        _plot_time_line(all_results, gammas, "Gamma (γ)",
                        "Calibración A: Tiempo vs Gamma",
                        "calA_gamma_time.png")

    print("\n→ Usa el gamma con mayor % éxito en el experimento principal.")
    return all_results


def calibration_episodes(save=False):
    """
    Exp B — Calibración de número de episodios.
    Solo Q-Learning y REINFORCE (agentes episódicos).
    Fija: 4x4, SR=0.8, gamma=config.GAMMA
    Varía: episodios = 500, 1000, 2000, 5000
    Objetivo: encontrar el número mínimo de episodios con el que convergen.
    Hipótesis: Q-Learning converge antes que REINFORCE por menor varianza.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN B — Número de episodios (agentes episódicos)")
    print("=" * 55)

    episode_counts = [500, 1000, 2000, 5000]
    map_name = "4x4"
    sr = 0.8
    gamma = config.GAMMA

    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]
    if not episodic_active:
        print("  (ningún agente episódico activo todavía — añade Q-Learning o REINFORCE)")
        return {}

    all_results = {}
    all_histories = {}
    for num_ep in episode_counts:
        print(f"\n  ep={num_ep} | Mapa={map_name} | SR={sr} | γ={gamma}")
        res, hist = run_agents(episodic_active, map_name, sr, gamma, num_ep,
                               save, f"calB_ep{num_ep}")
        all_results[num_ep] = res
        all_histories[num_ep] = hist

    if save:
        _plot_line(all_results, episode_counts, "Nº episodios",
                   "Calibración B: % Éxito vs Episodios",
                   "calB_episodes.png")
        # Curvas de aprendizaje con el máximo de episodios
        _plot_curves(all_histories[episode_counts[-1]],
                     f"Calibración B: Curvas ({episode_counts[-1]} ep.)",
                     f"calB_curves_{episode_counts[-1]}ep.png")

    print("\n→ Usa el mínimo de episodios donde la curva ya se estabiliza.")
    return all_results


# ── EXPERIMENTO PRINCIPAL ─────────────────────────────────────────────────────

def main_experiment(save=False):
    """
    Experimento principal — Mapa x Success Rate x Algoritmo.
    Fija: gamma y num_episodes justificados por la calibración.
    Varía: mapa (4x4, 8x8) x SR (0.33, 0.66, 1.0)

    Esto da una matriz completa de resultados que permite:
    - Ver cómo afecta SR para cada tamaño de mapa
    - Ver cómo escala cada algoritmo con el mapa
    - Identificar qué algoritmo es más robusto en cada escenario

    Hipótesis:
    - Value Iteration: óptimo en 4x4 pero escala mal en 8x8
    - Model Based: similar a VI pero más lento por la estimación
    - Q-Learning: escala mejor pero necesita más episodios en 8x8
    - REINFORCE: el más lento en converger, más afectado por SR bajo
    - SR bajo afecta más a los agentes que aprenden de experiencia (QL, REINFORCE, MB)
      que a VI que lee el MDP real
    """
    print("\n" + "=" * 55)
    print("EXPERIMENTO PRINCIPAL — Mapa x Success Rate x Algoritmo")
    print("=" * 55)

    maps = ["4x4", "8x8"]
    success_rates = [0.33, 0.66, 1.0]
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    # Estructura: all_results[map_name][sr][agent_name]
    all_results = {m: {} for m in maps}
    all_histories = {m: {} for m in maps}

    for map_name in maps:
        for sr in success_rates:
            print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
            label = f"main_map{map_name}_sr{sr}"
            res, hist = run_agents(ACTIVE_AGENTS, map_name, sr, gamma,
                                   num_ep, save, label)
            all_results[map_name][sr] = res
            all_histories[map_name][sr] = hist

    if save:
        # Gráfica por mapa: % éxito vs SR para cada algoritmo
        for map_name in maps:
            _plot_sr_per_map(all_results[map_name], success_rates,
                             f"% Éxito vs SR — Mapa {map_name}",
                             f"main_success_map{map_name}.png")
            _plot_time_per_map(all_results[map_name], success_rates,
                               f"Tiempo vs SR — Mapa {map_name}",
                               f"main_time_map{map_name}.png")

        # Gráfica comparativa 4x4 vs 8x8 para cada SR
        for sr in success_rates:
            _plot_map_comparison_bar(
                {m: all_results[m][sr] for m in maps},
                maps,
                f"4x4 vs 8x8 con SR={sr}",
                f"main_mapcomp_sr{sr}.png"
            )

        # Tabla resumen en CSV ya se guarda por run_agents
        # Tabla resumen impresa en pantalla
        _print_summary_table(all_results, maps, success_rates)

    return all_results, all_histories


# ── GRÁFICAS ──────────────────────────────────────────────────────────────────

def _plot_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["success"] for x in x_values]
        ax.plot(x_values, vals, marker="o", label=agent, color=COLORS.get(agent))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% Éxito")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["time"] for x in x_values]
        ax.plot(x_values, vals, marker="s", label=agent, color=COLORS.get(agent))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_curves(histories_by_agent, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, history in histories_by_agent.items():
        if not history:
            continue
        color = COLORS.get(name, "gray")
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


def _plot_sr_per_map(results_by_sr, success_rates, title, filename):
    """Líneas: un punto por SR, una línea por algoritmo."""
    agent_names = list(results_by_sr[success_rates[0]].keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    for ax, metric, ylabel in zip(
        axes,
        ["success", "time"],
        ["% Éxito", "Tiempo (s)"]
    ):
        for agent in agent_names:
            vals = [results_by_sr[sr][agent][metric] for sr in success_rates]
            ax.plot(success_rates, vals, marker="o",
                    label=agent, color=COLORS.get(agent))
        ax.set_xlabel("Success Rate")
        ax.set_ylabel(ylabel)
        ax.set_xticks(success_rates)
        ax.legend()
        ax.grid(True)
        if metric == "success":
            ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_per_map(results_by_sr, success_rates, title, filename):
    agent_names = list(results_by_sr[success_rates[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [results_by_sr[sr][agent]["time"] for sr in success_rates]
        ax.plot(success_rates, vals, marker="s",
                label=agent, color=COLORS.get(agent))
    ax.set_title(title)
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Tiempo (s)")
    ax.set_xticks(success_rates)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_map_comparison_bar(results_by_map, maps, title, filename):
    """Barras agrupadas: 4x4 vs 8x8 por algoritmo."""
    agent_names = list(results_by_map[maps[0]].keys())
    x = np.arange(len(agent_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(maps):
        vals = [results_by_map[m][a]["success"] for a in agent_names]
        bars = ax.bar(x + i * width, vals, width, label=f"Mapa {m}",
                      alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=8)

    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(agent_names)
    ax.set_ylabel("% Éxito")
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _print_summary_table(all_results, maps, success_rates):
    """Imprime tabla resumen en consola."""
    print("\n" + "=" * 70)
    print("TABLA RESUMEN — % Éxito")
    print("=" * 70)
    agent_names = list(all_results[maps[0]][success_rates[0]].keys())
    header = f"{'Mapa':<6} {'SR':<6} " + " ".join(f"{a:<18}" for a in agent_names)
    print(header)
    print("-" * 70)
    for m in maps:
        for sr in success_rates:
            row = f"{m:<6} {sr:<6} "
            for a in agent_names:
                pct = all_results[m][sr][a]["success"]
                row += f"{pct:<18.1f}"
            print(row)
    print("=" * 70)


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
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "calibration_gamma", "calibration_episodes", "main"],
        help="Experimento a ejecutar"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guarda gráficas y CSV en results/"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_dirs()

    if args.save:
        _init_csv()

    runners = {
        "calibration_gamma":     calibration_gamma,
        "calibration_episodes":  calibration_episodes,
        "main":                  main_experiment,
    }

    if args.exp == "all":
        for fn in runners.values():
            fn(save=args.save)
    else:
        runners[args.exp](save=args.save)

    if args.save:
        print(f"\nResultados en '{RESULTS_DIR}/' | CSV en '{CSV_PATH}'")