# experiments.py — batería completa de experimentos
#
# Uso:
#   python experiments.py --exp calibration_gamma       # calibración gamma
#   python experiments.py --exp calibration_episodes    # calibración episodios
#   python experiments.py --exp calibration_reward      # calibración reward shaping
#   python experiments.py --exp calibration_epsilon     # calibración epsilon y alpha
#   python experiments.py --exp main                    # experimento principal
#   python experiments.py --exp all --save              # todos + guarda resultados

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

ACTIVE_AGENTS = [
    "Value Iteration",
    "REINFORCE",
    "Q-Learning",
    "Model Based",
]

EPISODE_DEPENDENT = {"REINFORCE", "Q-Learning"}

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
    """Entrena y evalúa cada agente con la configuración dada."""
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
    Todos los algoritmos.
    Objetivo: justificar el gamma del experimento principal.
    Hipótesis: gamma alto funciona mejor porque la meta está lejos
               y las rutas óptimas son largas.
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
    Varía episodios = [500, 1000, 2000, 5000].
    Solo agentes episódicos: Q-Learning y REINFORCE.
    Fija: 4x4, SR=0.8, gamma=config.GAMMA.
    Objetivo: encontrar el mínimo de episodios con el que convergen.
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


# ── CALIBRACIÓN C — Reward shaping ───────────────────────────────────────────

def calibration_reward(save=False):
    """
    Compara tres señales de recompensa:
      - default:      recompensa original de Gymnasium (0 siempre, 1 en meta)
      - hole_penalty: penalización por caer en hoyo (HOLE_PENALTY = -0.5)
      - step_penalty: penalización por cada paso   (STEP_PENALTY = -0.01)
    Solo Q-Learning y REINFORCE porque VI y MB leen el MDP directamente
    y el wrapper no les afecta de la misma manera.
    Fija: 4x4, SR=0.8, gamma=config.GAMMA.
    Hipótesis: penalizar el hoyo ayuda a los agentes episódicos porque
               da señal negativa inmediata en lugar de solo silencio.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN C — Señal de recompensa")
    print("=" * 55)

    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]
    if not episodic_active:
        print("  (ningún agente episódico activo todavía)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES

    reward_configs = {
        "default":      {"DEFAULT_REWARD": True,  "HOLE_PENALTY": 0.0,  "STEP_PENALTY": 0.0},
        "hole_penalty": {"DEFAULT_REWARD": False, "HOLE_PENALTY": -0.5, "STEP_PENALTY": 0.0},
        "step_penalty": {"DEFAULT_REWARD": False, "HOLE_PENALTY": 0.0,  "STEP_PENALTY": -0.01},
    }

    all_results = {}
    all_histories = {}

    for reward_name, reward_cfg in reward_configs.items():
        for k, v in reward_cfg.items():
            setattr(config, k, v)

        print(f"\n  Reward={reward_name} | Mapa={map_name} | SR={sr}")
        res, hist = run_agents(episodic_active, map_name, sr, gamma, num_ep,
                               save, f"calC_{reward_name}")
        all_results[reward_name] = res
        all_histories[reward_name] = hist

    # Restaurar config
    config.DEFAULT_REWARD = False
    config.HOLE_PENALTY = -0.5
    config.STEP_PENALTY = 0.0

    if save:
        reward_labels = list(reward_configs.keys())
        fig, axes = plt.subplots(1, len(episodic_active),
                                 figsize=(6 * len(episodic_active), 5))
        if len(episodic_active) == 1:
            axes = [axes]
        for ax, agent in zip(axes, episodic_active):
            vals = [all_results[r][agent]["success"] for r in reward_labels]
            bars = ax.bar(reward_labels, vals,
                          color=["steelblue", "tomato", "seagreen"])
            ax.set_title(agent)
            ax.set_ylabel("% Éxito")
            ax.set_ylim(0, 110)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", fontsize=9)
            ax.grid(axis="y")
        fig.suptitle("Cal. C: % Éxito por señal de recompensa")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "calC_reward.png"))
        plt.close()

        for reward_name in reward_labels:
            _plot_curves(all_histories[reward_name],
                         f"Cal. C: Curvas — {reward_name}",
                         f"calC_curves_{reward_name}.png")

    print("\n→ Usa la señal con mejor % éxito en el experimento principal.")
    return all_results


## CALIBRACIÓN DE EPSILON Y ALPHA (solo Q-Learning)

def calibration_epsilon(save=False):
    """
    Varía epsilon inicial y alpha para Q-Learning.
    Fija: 4x4, SR=0.8, gamma=config.GAMMA, ep=config.NUM_EPISODES.
    Objetivo: justificar los valores de epsilon y alpha del experimento principal.
    Hipótesis: epsilon alto al inicio favorece la exploración pero necesita
               un decay suficiente para estabilizar la política. Alpha alto
               aprende más rápido pero puede oscilar más.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN D — Epsilon y Alpha (Q-Learning)")
    print("=" * 55)

    if "Q-Learning" not in ACTIVE_AGENTS:
        print("  (Q-Learning no está activo todavía)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES

    # Variamos epsilon inicial manteniendo alpha fijo
    epsilons = [0.1, 0.2, 0.5, 0.8]
    # Variamos alpha manteniendo epsilon fijo
    alphas = [0.05, 0.1, 0.3, 0.5]

    # ── Experimento D1: variación de epsilon ──────────────────────────────────
    print("\n  D1 — Variando epsilon (alpha fijo en config.LEARNING_RATE)")
    results_eps = {}
    histories_eps = {}
    original_epsilon = config.EPSILON

    for eps in epsilons:
        config.EPSILON = eps
        print(f"\n  ε={eps} | α={config.LEARNING_RATE} | Mapa={map_name} | SR={sr}")
        env = create_env(map_name=map_name, success_rate=sr)
        agent = build_agent("qlearning", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_eps[eps] = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_eps[eps] = history
        env.close()
        if save:
            _append_csv(f"calD1_eps{eps}", map_name, sr, gamma,
                        num_ep, "Q-Learning", pct, avg_r, elapsed)

    config.EPSILON = original_epsilon

    # ── Experimento D2: variación de alpha ────────────────────────────────────
    print("\n  D2 — Variando alpha (epsilon fijo en config.EPSILON)")
    results_alpha = {}
    histories_alpha = {}
    original_lr = config.LEARNING_RATE

    for alpha in alphas:
        config.LEARNING_RATE = alpha
        print(f"\n  ε={config.EPSILON} | α={alpha} | Mapa={map_name} | SR={sr}")
        env = create_env(map_name=map_name, success_rate=sr)
        agent = build_agent("qlearning", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_alpha[alpha] = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_alpha[alpha] = history
        env.close()
        if save:
            _append_csv(f"calD2_alpha{alpha}", map_name, sr, gamma,
                        num_ep, "Q-Learning", pct, avg_r, elapsed)

    config.LEARNING_RATE = original_lr

    if save:
        _plot_line(results_eps, epsilons, "Epsilon inicial (ε)",
                   "Cal. D1: % Éxito vs Epsilon (Q-Learning)",
                   "calD1_epsilon.png")
        _plot_line(results_alpha, alphas, "Learning rate (α)",
                   "Cal. D2: % Éxito vs Alpha (Q-Learning)",
                   "calD2_alpha.png")
        _plot_curves(histories_eps[epsilons[-1]],
                     f"Cal. D1: Curva con ε={epsilons[-1]}",
                     f"calD1_curves_eps{epsilons[-1]}.png")
        _plot_curves(histories_alpha[alphas[-1]],
                     f"Cal. D2: Curva con α={alphas[-1]}",
                     f"calD2_curves_alpha{alphas[-1]}.png")

    print("\n→ Usa el epsilon y alpha con mejor % éxito en el experimento principal.")
    return {"epsilon": results_eps, "alpha": results_alpha}


## EXPERIMENTO PRINCIPAL

def main_experiment(save=False):
    """
    Experimento principal: Mapa x SR x Algoritmo.
    Fija: gamma y num_episodes justificados por calibraciones A y B.
    Varía: mapa (4x4, 8x8) x SR (0.33, 0.66, 1.0).
    Hipótesis:
    - VI: óptimo en 4x4, escala mal en 8x8 por coste del MDP completo
    - MB: similar a VI pero más lento por la estimación desde experiencia
    - QL: escala mejor, necesita más episodios en 8x8
    - REINFORCE: más lento en converger, más afectado por SR bajo
    - SR bajo afecta más a QL/REINFORCE/MB que a VI porque VI lee el MDP real
    Métricas: % éxito, tiempo de entrenamiento, curvas de aprendizaje.
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
        # % Éxito y tiempo vs SR por mapa
        for map_name in maps:
            _plot_sr_per_map(all_results[map_name], success_rates,
                             f"% Éxito y Tiempo vs SR — {map_name}",
                             f"main_sr_{map_name}.png")

        # 4x4 vs 8x8 para cada SR
        for sr in success_rates:
            _plot_map_comparison_bar(
                {m: all_results[m][sr] for m in maps}, maps,
                f"4x4 vs 8x8 — SR={sr}",
                f"main_mapcomp_sr{sr}.png"
            )

        # Curvas de aprendizaje para agentes episódicos
        for map_name in maps:
            for sr in success_rates:
                episodic_hist = {k: v for k, v in all_histories[map_name][sr].items()
                                 if k in EPISODE_DEPENDENT and v}
                if episodic_hist:
                    _plot_curves(episodic_hist,
                                 f"Curvas — {map_name} SR={sr}",
                                 f"main_curves_{map_name}_sr{sr}.png")

        _print_summary_table(all_results, maps, success_rates)
        _print_time_table(all_results, maps, success_rates)

    return all_results, all_histories


## GENERAR GRÁFICAS Y PLOTS

def _plot_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        color = COLORS.get(agent, "gray")
        vals = [all_results[x][agent]["success"] for x in x_values]
        ax.plot(x_values, vals, marker="o", label=agent, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="% Éxito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        color = COLORS.get(agent, "gray")
        vals = [all_results[x][agent]["time"] for x in x_values]
        ax.plot(x_values, vals, marker="s", label=agent, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
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
        else:
            smooth = history
        ax.plot(smooth, label=name, color=color)
    ax.set(title=title, xlabel="Episodios", ylabel="Recompensa")
    ax.legend()
    ax.grid(True)
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
            ax.plot(success_rates, vals, marker="o",
                    label=agent, color=COLORS.get(agent, "gray"))
        ax.set(xlabel="Success Rate", ylabel=ylabel)
        ax.set_xticks(success_rates)
        if metric == "success":
            ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True)
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
    ax.legend()
    ax.grid(axis="y")
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


def _print_time_table(all_results, maps, success_rates):
    print("\n" + "=" * 70)
    print("TABLA RESUMEN — Tiempo de entrenamiento (s)")
    print("=" * 70)
    agent_names = list(all_results[maps[0]][success_rates[0]].keys())
    print(f"{'Mapa':<6} {'SR':<6} " + " ".join(f"{a:<18}" for a in agent_names))
    print("-" * 70)
    for m in maps:
        for sr in success_rates:
            row = f"{m:<6} {sr:<6} "
            for a in agent_names:
                row += f"{all_results[m][sr][a]['time']:<18.2f}"
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
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["all", "calibration_gamma", "calibration_episodes",
                 "calibration_reward", "calibration_epsilon", "main"],
        help="Experimento a ejecutar. 'all' ejecuta todos en orden."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guarda gráficas y CSV en results/"
    )
    args = parser.parse_args()

    setup_dirs()
    if args.save:
        _init_csv()

    runners = {
        "calibration_gamma":    calibration_gamma,
        "calibration_episodes": calibration_episodes,
        "calibration_reward":   calibration_reward,
        "calibration_epsilon":  calibration_epsilon,
        "main":                 main_experiment,
    }

    if args.exp == "all":
        for fn in runners.values():
            fn(save=args.save)
    else:
        runners[args.exp](save=args.save)

    if args.save:
        print(f"\nResultados en '{RESULTS_DIR}/' | CSV en '{CSV_PATH}'")