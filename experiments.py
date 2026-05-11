# experiments.py — batería completa de experimentos
#
# Uso:
#   python experiments.py --exp calibration_gamma
#   python experiments.py --exp calibration_episodes
#   python experiments.py --exp calibration_reward
#   python experiments.py --exp calibration_epsilon
#   python experiments.py --exp calibration_reinforce_lr
#   python experiments.py --exp calibration_transitions
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
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
CSV_PATH    = os.path.join(RESULTS_DIR, "results.csv")

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

# Semillas fijas para reproducibilidad
# En calibraciones se usa solo SEEDS[0]; en el experimento principal las 3
SEEDS = [42, 123, 7]


def setup_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Entrenamiento y evaluación ────────────────────────────────────────────────

def _set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_agent(agent):
    start   = time.time()
    result  = agent.train()
    elapsed = time.time() - start
    history = result if isinstance(result, list) else []
    return history, elapsed


def run_single(agent_name, map_name, success_rate, gamma, num_episodes, seed):
    """Entrena y evalúa un agente con una semilla concreta. Devuelve dict de métricas."""
    _set_seed(seed)
    config.GAMMA        = gamma
    config.NUM_EPISODES = num_episodes

    env     = create_env(map_name=map_name, success_rate=success_rate)
    agent   = build_agent(AGENT_KEYS[agent_name], env)
    history, elapsed = train_agent(agent)
    pct, avg_r = evaluate_agent(agent, env)
    env.close()
    return {"success": pct, "avg_reward": avg_r, "time": elapsed, "history": history}


def run_agents(agents_to_run, map_name, success_rate, gamma,
               num_episodes, save, label, seeds=None):
    """
    Entrena y evalúa cada agente.
    Si seeds tiene más de un valor, promedia los resultados y calcula std.
    """
    if seeds is None:
        seeds = [SEEDS[0]]

    config.GAMMA        = gamma
    config.NUM_EPISODES = num_episodes

    results   = {}
    histories = {}

    for name in agents_to_run:
        print(f"    [{name}] entrenando ({len(seeds)} semilla/s)...", end=" ", flush=True)

        runs = [run_single(name, map_name, success_rate, gamma, num_episodes, s)
                for s in seeds]

        pct_vals  = [r["success"]    for r in runs]
        time_vals = [r["time"]       for r in runs]
        avg_vals  = [r["avg_reward"] for r in runs]

        results[name] = {
            "success":     np.mean(pct_vals),
            "success_std": np.std(pct_vals),
            "avg_reward":  np.mean(avg_vals),
            "time":        np.mean(time_vals),
            "history":     runs[0]["history"],   # curva de la primera semilla
        }
        histories[name] = results[name]["history"]

        print(f"éxito={results[name]['success']:.1f}% "
              f"(±{results[name]['success_std']:.1f}) | "
              f"tiempo={results[name]['time']:.2f}s")

        if save:
            _append_csv(label, map_name, success_rate, gamma, num_episodes,
                        name, results[name]["success"], results[name]["avg_reward"],
                        results[name]["time"])

    return results, histories


# ── CALIBRACIÓN A — Gamma ─────────────────────────────────────────────────────

def calibration_gamma(save=False):
    """
    Varía gamma = [0.90, 0.95, 0.99].
    Fija: 4x4, SR=1/3 (máxima estocasticidad), semilla fija.
    Todos los algoritmos.
    Hipótesis: gamma alto funciona mejor porque las rutas óptimas son largas.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN A — Efecto de gamma")
    print("=" * 55)

    gammas   = [0.90, 0.95, 0.99]
    map_name = "4x4"
    sr       = 1 / 3
    num_ep   = config.NUM_EPISODES

    all_results = {}
    for g in gammas:
        print(f"\n  γ={g} | Mapa={map_name} | SR={sr:.2f}")
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
    Solo Q-Learning y REINFORCE. Fija: 4x4, SR=0.8, gamma=config.GAMMA.
    Hipótesis: Q-Learning converge antes que REINFORCE por menor varianza.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN B — Número de episodios")
    print("=" * 55)

    episode_counts  = [500, 1000, 2000, 5000]
    map_name, sr, gamma = "4x4", 0.8, config.GAMMA
    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]

    if not episodic_active:
        print("  (ningún agente episódico activo)")
        return {}

    all_results   = {}
    all_histories = {}
    for num_ep in episode_counts:
        print(f"\n  ep={num_ep} | Mapa={map_name} | SR={sr} | γ={gamma}")
        res, hist = run_agents(episodic_active, map_name, sr, gamma, num_ep,
                               save, f"calB_ep{num_ep}")
        all_results[num_ep]   = res
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
      - default:      recompensa original (1 meta, 0 resto)
      - hole_penalty: penalización por hoyo (-1)
      - step_penalty: penalización por paso (-0.01)
    Solo Q-Learning y REINFORCE. Fija: 4x4, SR=0.8.
    Hipótesis: penalizar el hoyo da señal negativa inmediata y ayuda a converger.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN C — Señal de recompensa")
    print("=" * 55)

    episodic_active = [a for a in ACTIVE_AGENTS if a in EPISODE_DEPENDENT]
    if not episodic_active:
        print("  (ningún agente episódico activo)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES

    reward_configs = {
        "default":      {"DEFAULT_REWARD": True,  "HOLE_PENALTY":  0.0,  "STEP_PENALTY":  0.0},
        "hole_penalty": {"DEFAULT_REWARD": False, "HOLE_PENALTY": -1.0,  "STEP_PENALTY":  0.0},
        "step_penalty": {"DEFAULT_REWARD": False, "HOLE_PENALTY":  0.0,  "STEP_PENALTY": -0.01},
    }

    all_results   = {}
    all_histories = {}

    for reward_name, reward_cfg in reward_configs.items():
        for k, v in reward_cfg.items():
            setattr(config, k, v)
        print(f"\n  Reward={reward_name} | Mapa={map_name} | SR={sr}")
        res, hist = run_agents(episodic_active, map_name, sr, gamma, num_ep,
                               save, f"calC_{reward_name}")
        all_results[reward_name]   = res
        all_histories[reward_name] = hist

    # Restaurar config
    config.DEFAULT_REWARD = False
    config.HOLE_PENALTY   = -0.5
    config.STEP_PENALTY   = 0.0

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


# ── CALIBRACIÓN D — Epsilon y Alpha (Q-Learning) ─────────────────────────────

def calibration_epsilon(save=False):
    """
    D1: varía epsilon ∈ {0.1, 0.3, 0.5, 0.8, 1.0} con alpha fijo.
    D2: varía alpha ∈ {0.05, 0.1, 0.3, 0.5} con el epsilon ganador de D1.
    Solo Q-Learning. Fija: 4x4, SR=0.8.
    Hipótesis: epsilon muy bajo converge a subóptimo, muy alto tarda más.
               Alpha alto aprende rápido pero puede oscilar.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN D — Epsilon y Alpha (Q-Learning)")
    print("=" * 55)

    if "Q-Learning" not in ACTIVE_AGENTS:
        print("  (Q-Learning no está activo)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES
    epsilons = [0.1, 0.3, 0.5, 0.8, 1.0]
    alphas   = [0.05, 0.1, 0.3, 0.5]

    original_epsilon = config.EPSILON
    original_lr      = config.LEARNING_RATE

    # D1 — variación de epsilon
    print("\n  D1 — Variando epsilon (alpha fijo)")
    results_eps   = {}
    histories_eps = {}
    for eps in epsilons:
        config.EPSILON = eps
        print(f"\n  ε={eps} | α={config.LEARNING_RATE} | Mapa={map_name} | SR={sr}")
        env     = create_env(map_name=map_name, success_rate=sr)
        agent   = build_agent("qlearning", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        env.close()
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_eps[eps]   = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_eps[eps] = history
        if save:
            _append_csv(f"calD1_eps{eps}", map_name, sr, gamma,
                        num_ep, "Q-Learning", pct, avg_r, elapsed)

    config.EPSILON = original_epsilon

    # D2 — variación de alpha
    print("\n  D2 — Variando alpha (epsilon fijo)")
    results_alpha   = {}
    histories_alpha = {}
    for alpha in alphas:
        config.LEARNING_RATE = alpha
        print(f"\n  ε={config.EPSILON} | α={alpha} | Mapa={map_name} | SR={sr}")
        env     = create_env(map_name=map_name, success_rate=sr)
        agent   = build_agent("qlearning", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        env.close()
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_alpha[alpha]   = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_alpha[alpha] = history
        if save:
            _append_csv(f"calD2_alpha{alpha}", map_name, sr, gamma,
                        num_ep, "Q-Learning", pct, avg_r, elapsed)

    config.LEARNING_RATE = original_lr

    if save:
        _plot_line_single(results_eps, epsilons, "Epsilon inicial (ε)",
                          "Cal. D1: % Éxito vs Epsilon (Q-Learning)",
                          "calD1_epsilon.png", "Q-Learning")
        _plot_line_single(results_alpha, alphas, "Learning rate (α)",
                          "Cal. D2: % Éxito vs Alpha (Q-Learning)",
                          "calD2_alpha.png", "Q-Learning")
        _plot_curves({f"ε={epsilons[-1]}": histories_eps[epsilons[-1]]},
                     f"Cal. D1: Curva ε={epsilons[-1]}",
                     f"calD1_curves_eps{epsilons[-1]}.png")
        _plot_curves({f"α={alphas[-1]}": histories_alpha[alphas[-1]]},
                     f"Cal. D2: Curva α={alphas[-1]}",
                     f"calD2_curves_alpha{alphas[-1]}.png")

    print("\n→ Usa el epsilon y alpha con mejor % éxito en el experimento principal.")
    return {"epsilon": results_eps, "alpha": results_alpha}


# ── CALIBRACIÓN E — Learning rate REINFORCE ───────────────────────────────────

def calibration_reinforce_lr(save=False):
    """
    Varía REINFORCE_LR ∈ {0.001, 0.01, 0.05, 0.1}.
    Fija: 4x4, SR=0.8, gamma=config.GAMMA, episodios=config.NUM_EPISODES.
    Solo REINFORCE.
    Hipótesis: LR muy alto hace el gradiente inestable; muy bajo converge lento.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN E — Learning rate REINFORCE")
    print("=" * 55)

    if "REINFORCE" not in ACTIVE_AGENTS:
        print("  (REINFORCE no está activo)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES
    lrs = [0.001, 0.01, 0.05, 0.1]
    original_lr = config.REINFORCE_LR

    results_lr   = {}
    histories_lr = {}

    for lr in lrs:
        config.REINFORCE_LR = lr
        print(f"\n  LR={lr} | Mapa={map_name} | SR={sr} | γ={gamma}")
        env     = create_env(map_name=map_name, success_rate=sr)
        agent   = build_agent("reinforce", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        env.close()
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_lr[lr]   = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_lr[lr] = history
        if save:
            _append_csv(f"calE_lr{lr}", map_name, sr, gamma,
                        num_ep, "REINFORCE", pct, avg_r, elapsed)

    config.REINFORCE_LR = original_lr

    if save:
        _plot_line_single(results_lr, lrs, "Learning rate (α)",
                          "Cal. E: % Éxito vs LR (REINFORCE)",
                          "calE_reinforce_lr.png", "REINFORCE")
        _plot_curves({f"lr={lrs[-1]}": histories_lr[lrs[-1]]},
                     f"Cal. E: Curva LR={lrs[-1]}",
                     f"calE_curves_lr{lrs[-1]}.png")

    print("\n→ Usa el LR con mejor % éxito en el experimento principal.")
    return results_lr


# ── CALIBRACIÓN F — Transiciones muestreadas (Model Based) ───────────────────

def calibration_transitions(save=False):
    """
    Varía transiciones por iteración ∈ {100, 500, 1000, 5000}.
    Solo Model Based. Fija: 4x4, SR=0.8, gamma=config.GAMMA.
    Hipótesis: pocas transiciones dan estimaciones imprecisas del MDP;
               muchas aumentan el tiempo sin mejorar el resultado.
    """
    print("\n" + "=" * 55)
    print("CALIBRACIÓN F — Transiciones muestreadas (Model Based)")
    print("=" * 55)

    if "Model Based" not in ACTIVE_AGENTS:
        print("  (Model Based no está activo)")
        return {}

    map_name, sr, gamma, num_ep = "4x4", 0.8, config.GAMMA, config.NUM_EPISODES
    transitions = [100, 500, 1000, 5000]

    # Model Based usa NUM_TRAJECTORIES si está en config, si no lo añadimos
    if not hasattr(config, "NUM_TRAJECTORIES"):
        config.NUM_TRAJECTORIES = 100
    original_traj = config.NUM_TRAJECTORIES

    results_traj   = {}
    histories_traj = {}

    for traj in transitions:
        config.NUM_TRAJECTORIES = traj
        print(f"\n  traj={traj} | Mapa={map_name} | SR={sr} | γ={gamma}")
        env     = create_env(map_name=map_name, success_rate=sr)
        agent   = build_agent("model_based", env)
        history, elapsed = train_agent(agent)
        pct, avg_r = evaluate_agent(agent, env)
        env.close()
        print(f"    éxito={pct:.1f}% | tiempo={elapsed:.2f}s")
        results_traj[traj]   = {"success": pct, "avg_reward": avg_r, "time": elapsed}
        histories_traj[traj] = history
        if save:
            _append_csv(f"calF_traj{traj}", map_name, sr, gamma,
                        num_ep, "Model Based", pct, avg_r, elapsed)

    config.NUM_TRAJECTORIES = original_traj

    if save:
        _plot_line_single(results_traj, transitions,
                          "Transiciones por iteración",
                          "Cal. F: % Éxito vs Transiciones (Model Based)",
                          "calF_transitions.png", "Model Based")
        _plot_line_single_time(results_traj, transitions,
                               "Transiciones por iteración",
                               "Cal. F: Tiempo vs Transiciones (Model Based)",
                               "calF_transitions_time.png", "Model Based")

    print("\n→ Usa el mínimo de transiciones con el que el % éxito se estabiliza.")
    return results_traj


# ── EXPERIMENTO PRINCIPAL ─────────────────────────────────────────────────────

def main_experiment(save=False):
    """
    Mapa (4x4, 8x8) x SR (0.33, 0.66, 1.0) x Algoritmo.
    Usa 3 semillas y reporta media ± std para rigor estadístico.
    Hipótesis:
    - VI: óptimo en 4x4, escala mal en 8x8 (coste del MDP completo)
    - MB: similar a VI pero más lento por la estimación desde experiencia
    - QL: escala mejor, necesita más episodios en 8x8
    - REINFORCE: más lento en converger, más afectado por SR bajo
    - SR bajo afecta más a QL/REINFORCE/MB que a VI (lee el MDP real)
    """
    print("\n" + "=" * 55)
    print("EXPERIMENTO PRINCIPAL — Mapa x SR x Algoritmo (3 semillas)")
    print("=" * 55)

    maps          = ["4x4", "8x8"]
    success_rates = [0.33, 0.66, 1.0]
    gamma         = config.GAMMA
    num_ep        = config.NUM_EPISODES

    all_results   = {m: {} for m in maps}
    all_histories = {m: {} for m in maps}

    for map_name in maps:
        for sr in success_rates:
            print(f"\n  Mapa={map_name} | SR={sr} | γ={gamma} | ep={num_ep}")
            res, hist = run_agents(ACTIVE_AGENTS, map_name, sr, gamma,
                                   num_ep, save,
                                   f"main_{map_name}_sr{sr}",
                                   seeds=SEEDS)
            all_results[map_name][sr]   = res
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


# ── GRÁFICAS ──────────────────────────────────────────────────────────────────

def _plot_line(all_results, x_values, xlabel, title, filename):
    """Gráfica de líneas con múltiples agentes."""
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        color = COLORS.get(agent, "gray")
        vals  = [all_results[x][agent]["success"] for x in x_values]
        stds  = [all_results[x][agent].get("success_std", 0) for x in x_values]
        ax.plot(x_values, vals, marker="o", label=agent, color=color)
        ax.fill_between(x_values,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.15, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="% Éxito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_line_single(results, x_values, xlabel, title, filename, agent_name):
    """Gráfica de línea para un único agente (calibraciones D, E, F)."""
    color = COLORS.get(agent_name, "gray")
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[x]["success"] for x in x_values]
    ax.plot(x_values, vals, marker="o", color=color, label=agent_name)
    ax.set(title=title, xlabel=xlabel, ylabel="% Éxito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_line_single_time(results, x_values, xlabel, title, filename, agent_name):
    """Gráfica de tiempo para un único agente."""
    color = COLORS.get(agent_name, "gray")
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[x]["time"] for x in x_values]
    ax.plot(x_values, vals, marker="s", color=color, label=agent_name)
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_line(all_results, x_values, xlabel, title, filename):
    """Gráfica de tiempo con múltiples agentes."""
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        color = COLORS.get(agent, "gray")
        vals  = [all_results[x][agent]["time"] for x in x_values]
        ax.plot(x_values, vals, marker="s", label=agent, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_curves(histories_by_agent, title, filename):
    """Curvas de aprendizaje suavizadas."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, history in histories_by_agent.items():
        if not history:
            continue
        color = COLORS.get(name, "gray")
        ax.plot(history, alpha=0.2, color=color)
        smooth = (np.convolve(history, np.ones(100) / 100, mode="valid")
                  if len(history) >= 100 else history)
        ax.plot(smooth, label=name, color=color)
    ax.set(title=title, xlabel="Episodios", ylabel="Recompensa")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_sr_per_map(results_by_sr, success_rates, title, filename):
    """% Éxito y Tiempo vs SR para un mapa concreto."""
    agent_names = list(results_by_sr[success_rates[0]].keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    for ax, metric, ylabel in zip(axes,
                                   ["success", "time"],
                                   ["% Éxito", "Tiempo (s)"]):
        for agent in agent_names:
            vals  = [results_by_sr[sr][agent][metric] for sr in success_rates]
            stds  = ([results_by_sr[sr][agent].get("success_std", 0)
                      for sr in success_rates]
                     if metric == "success" else [0] * len(success_rates))
            color = COLORS.get(agent, "gray")
            ax.plot(success_rates, vals, marker="o", label=agent, color=color)
            if metric == "success":
                ax.fill_between(success_rates,
                                [v - s for v, s in zip(vals, stds)],
                                [v + s for v, s in zip(vals, stds)],
                                alpha=0.15, color=color)
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
    """Barras agrupadas 4x4 vs 8x8 por algoritmo."""
    agent_names = list(results_by_map[maps[0]].keys())
    x     = np.arange(len(agent_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(maps):
        vals = [results_by_map[m][a]["success"] for a in agent_names]
        stds = [results_by_map[m][a].get("success_std", 0) for a in agent_names]
        bars = ax.bar(x + i * width, vals, width, label=f"Mapa {m}",
                      alpha=0.8, yerr=stds, capsize=4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}%", ha="center", fontsize=8)
    ax.set(title=title, ylabel="% Éxito", ylim=(0, 120))
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(agent_names)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _print_summary_table(all_results, maps, success_rates):
    print("\n" + "=" * 75)
    print("TABLA RESUMEN — % Éxito (media ± std)")
    print("=" * 75)
    agent_names = list(all_results[maps[0]][success_rates[0]].keys())
    print(f"{'Mapa':<6} {'SR':<6} " + " ".join(f"{a:<22}" for a in agent_names))
    print("-" * 75)
    for m in maps:
        for sr in success_rates:
            row = f"{m:<6} {sr:<6} "
            for a in agent_names:
                pct = all_results[m][sr][a]["success"]
                std = all_results[m][sr][a].get("success_std", 0)
                row += f"{pct:.1f}±{std:.1f}{'':>10}"
            print(row)
    print("=" * 75)


def _print_time_table(all_results, maps, success_rates):
    print("\n" + "=" * 75)
    print("TABLA RESUMEN — Tiempo de entrenamiento (s)")
    print("=" * 75)
    agent_names = list(all_results[maps[0]][success_rates[0]].keys())
    print(f"{'Mapa':<6} {'SR':<6} " + " ".join(f"{a:<22}" for a in agent_names))
    print("-" * 75)
    for m in maps:
        for sr in success_rates:
            row = f"{m:<6} {sr:<6} "
            for a in agent_names:
                row += f"{all_results[m][sr][a]['time']:<22.2f}"
            print(row)
    print("=" * 75)


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
                 "calibration_reward", "calibration_epsilon",
                 "calibration_reinforce_lr", "calibration_transitions", "main"],
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
        "calibration_gamma":        calibration_gamma,
        "calibration_episodes":     calibration_episodes,
        "calibration_reward":       calibration_reward,
        "calibration_epsilon":      calibration_epsilon,
        "calibration_reinforce_lr": calibration_reinforce_lr,
        "calibration_transitions":  calibration_transitions,
        "main":                     main_experiment,
    }

    if args.exp == "all":
        for fn in runners.values():
            fn(save=args.save)
    else:
        runners[args.exp](save=args.save)

    if args.save:
        print(f"\nResultados en '{RESULTS_DIR}/' | CSV en '{CSV_PATH}'")