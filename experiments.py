# experiments.py — batería completa de experimentos
#
# Uso:
#   python experiments.py --exp calibration_gamma
#   python experiments.py --exp calibration_episodes
#   python experiments.py --exp main
#   python experiments.py --exp all --save

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import config
from main import create_env, evaluate_agent, build_agent

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

# Descomenta cuando tus compañeros entreguen sus ficheros
ACTIVE_AGENTS = [
    "Value Iteration",
    "REINFORCE",
    # "Q-Learning",  # [PENDING]
    # "Model Based", # [PENDING]
]

EPISODE_DEPENDENT = {"REINFORCE", "Q-Learning"}

# Mapeo nombre legible → clave de build_agent
AGENT_KEYS = {
    "Value Iteration": "value_iteration",
    "REINFORCE":       "reinforce",
    "Q-Learning":      "qlearning",
    "Model Based":     "model_based",
}


def setup_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Entrenamiento y evaluación ────────────────────────────────────────────────

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
        agent = build_agent(AGENT_KEYS[name], env)
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


# ── CALIBRACIÓN A — Gamma ─────────────────────────────────────────────────────

def calibration_gamma(save=False):
    """
    Varía gamma = [0.90, 0.95, 0.99]. Fija: 4x4, SR=0.8.
    Objetivo: justificar el gamma del experimento principal.
    Hipótesis: gamma alto funciona mejor porque las rutas óptimas son largas.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN A — Efecto de gamma")
    print("=" * 55)

    gammas = [0.90, 0.95, 0.99]
    map_name, sr, num_ep = "4x4", 0.8, config.NUM_EPISODES

    all_results = {}
    for g in gammas:
        print(f"\n  γ={g} | Mapa={map_name} | SR={sr}")
        res, _ = run_agents(ACTIVE_AGENTS, map_name, sr, g, num_ep,
                            save, f"calA_g{g}")
        all_results[g] = res

    if save:
        _plot_line(all_results, gammas, "Gamma (γ)",
                   "Cal. A: % Éxito vs Gamma", "calA_gamma.png")
        _plot_time_line(all_results, gammas, "Gamma (γ)",
                        "Cal. A: Tiempo vs Gamma", "calA_gamma_time.png")

    print("\n→ Usa el gamma con mayor % éxito en el experimento principal.")
    return all_results


# ── CALIBRACIÓN B — Episodios ─────────────────────────────────────────────────

def calibration_episodes(save=False):
    """
    Varía episodios = [500, 1000, 2000, 5000]. Solo agentes episódicos.
    Fija: 4x4, SR=0.8.
    Objetivo: mínimo de episodios con el que convergen.
    Hipótesis: Q-Learning converge antes que REINFORCE por menor varianza.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN B — Número de episodios")
    print("=" * 55)

    episode_counts = [500, 1000, 2000, 5000]
    map_name, sr, gamma = "4x4", 0.8, config.GAMMA

    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]
    if not episodic_active:
        print("  (ningún agente episódico activo todavía)")
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
                   "Cal. B: % Éxito vs Episodios", "calB_episodes.png")
        _plot_curves(all_histories[episode_counts[-1]],
                     f"Cal. B: Curvas ({episode_counts[-1]} ep.)",
                     f"calB_curves_{episode_counts[-1]}ep.png")

    print("\n→ Usa el mínimo donde la curva se estabiliza.")
    return all_results


# ── EXPERIMENTO PRINCIPAL ─────────────────────────────────────────────────────

def main_experiment(save=False):
    """
    Mapa (4x4, 8x8) x SR (0.33, 0.66, 1.0) x Algoritmo.
    Hipótesis:
    - VI: óptimo en 4x4, escala mal en 8x8
    - MB: similar a VI pero más lento
    - QL: escala mejor, necesita más episodios en 8x8
    - REINFORCE: más lento, más afectado por SR bajo
    - SR bajo afecta más a QL/REINFORCE/MB que a VI (lee el MDP real)
    """
    print("\n" + "=" * 55)
    print("EXPERIMENTO PRINCIPAL — Mapa x SR x Algoritmo")
    print("=" * 55)

    maps = ["4x4", "8x8"]
    success_rates = [0.33, 0.66, 1.0]
    gamma, num_ep = config.GAMMA, config.NUM_EPISODES

    all_results = {m: {} for m in maps}
    all_histories = {m: {} for m in maps}

    for map_name in maps:
        for sr in success_rates:
            print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
            res, hist = run_agents(ACTIVE_AGENTS, map_name, sr, gamma,
                                   num_ep, save, f"main_{map_name}_sr{sr}")
            all_results[map_name][sr] = res
            all_histories[map_name][sr] = hist

    if save:
        for map_name in maps:
            _plot_sr_per_map(all_results[map_name], success_rates,
                             f"% Éxito y Tiempo vs SR — {map_name}",
                             f"main_sr_{map_name}.png")
        for sr in success_rates:
            _plot_map_comparison_bar(
                {m: all_results[m][sr] for m in maps}, maps,
                f"4x4 vs 8x8 — SR={sr}",
                f"main_mapcomp_sr{sr}.png"
            )
        _print_summary_table(all_results, maps, success_rates)

    return all_results, all_histories


# ── GRÁFICAS ──────────────────────────────────────────────────────────────────

def _plot_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["success"] for x in x_values]
        ax.plot(x_values, vals, marker="o", label=agent, color=COLORS.get(agent))
    ax.set(title=title, xlabel=xlabel, ylabel="% Éxito", ylim=(0, 110))
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["time"] for x in x_values]
        ax.plot(x_values, vals, marker="s", label=agent, color=COLORS.get(agent))
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
    ax.legend(); ax.grid(True)
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
        smooth = np.convolve(history, np.ones(100) / 100, mode="valid") if len(history) >= 100 else history
        ax.plot(smooth, label=name, color=color)
    ax.set(title=title, xlabel="Episodios", ylabel="Recompensa")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_sr_per_map(results_by_sr, success_rates, title, filename):
    agent_names = list(results_by_sr[success_rates[0]].keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    for ax, metric, ylabel in zip(axes, ["success", "time"], ["% Éxito", "Tiempo (s)"]):
        for agent in agent_names:
            vals = [results_by_sr[sr][agent][metric] for sr in success_rates]
            ax.plot(success_rates, vals, marker="o", label=agent, color=COLORS.get(agent))
        ax.set(xlabel="Success Rate", ylabel=ylabel)
        ax.set_xticks(success_rates)
        if metric == "success":
            ax.set_ylim(0, 110)
        ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_map_comparison_bar(results_by_map, maps, title, filename):
    agent_names = list(results_by_map[maps[0]].keys())
    x = np.arange(len(agent_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(maps):
        vals = [results_by_map[m][a]["success"] for a in agent_names]
        bars = ax.bar(x + i * width, vals, width, label=f"Mapa {m}", alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=8)
    ax.set(title=title, ylabel="% Éxito", ylim=(0, 115))
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(agent_names)
    ax.legend(); ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _print_summary_table(all_results, maps, success_rates):
    print("\n" + "=" * 70)
    print("TABLA RESUMEN — % Éxito")
    print("=" * 70)
    agent_names = list(all_results[maps[0]][success_rates[0]].keys())
    print(f"{'Mapa':<6} {'SR':<6} " + " ".join(f"{a:<18}" for a in agent_names))
    print("-" * 70)
    for m in maps:
        for sr in success_rates:
            row = f"{m:<6} {sr:<6} "
            for a in agent_names:
                row += f"{all_results[m][sr][a]['success']:<18.1f}"
            print(row)
    print("=" * 70)


# ── CSV ───────────────────────────────────────────────────────────────────────

def _init_csv():
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["exp", "map_name", "success_rate", "gamma",
                                 "num_episodes", "agent", "pct_success",
                                 "avg_reward", "train_time_s"])


def _append_csv(exp, map_name, sr, gamma, num_ep, agent, pct, avg_r, elapsed):
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([exp, map_name, sr, gamma, num_ep, agent,
                                 f"{pct:.2f}", f"{avg_r:.4f}", f"{elapsed:.2f}"])


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimentos FrozenLake")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "calibration_gamma",
                                 "calibration_episodes", "main"])
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    setup_dirs()
    if args.save:
        _init_csv()

    runners = {
        "calibration_gamma":    calibration_gamma,
        "calibration_episodes": calibration_episodes,
        "main":                 main_experiment,
    }

    if args.exp == "all":
        for fn in runners.values():
            fn(save=args.save)
    else:
        runners[args.exp](save=args.save)

    if args.save:
        print(f"\nResultados en '{RESULTS_DIR}/' | CSV en '{CSV_PATH}'")