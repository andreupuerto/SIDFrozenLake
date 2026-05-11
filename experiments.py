# experiments.py
# Changes: replaced the old single-CSV/one-seed experiment runner with the
# dual-seed schema, per-run metrics, aggregation helpers, split CSV outputs,
# calibration updates, scaling/stochasticity experiments, and sanity checks
# required in Sections 3.1-3.10. The runner also consumes the new diagnostics
# from main.py (Sections 2.2-2.4) and the Model-Based budget from config.py
# (Section 4.1).

import argparse
import contextlib
import csv
import io
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np

import config
from main import (
    build_agent,
    compute_state_action_coverage,
    create_env,
    evaluate_agent,
    evaluate_agent_with_oracle,
)


# -- Directorios --------------------------------------------------------------

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

RESULTS_CSV_HEADER = [
    "exp_id", "algorithm", "map_name", "success_rate", "reward_schedule",
    "agent_seed", "env_seed", "map_seed",
    "hyperparams_json",
    "total_train_time_s", "n_train_episodes", "n_train_iterations",
    "eval_success_rate", "eval_mean_reward", "eval_mean_steps",
    "eval_termination_goal_pct", "eval_termination_hole_pct",
    "eval_truncation_pct",
    "eval_policy_agreement_vi", "n_state_action_visited_pct",
    "final_q_diff_l_inf",
]

CURVES_CSV_HEADER = [
    "exp_id", "algorithm", "agent_seed", "episode",
    "train_reward", "train_steps", "train_success",
    "epsilon", "alpha", "elapsed_time_s",
]

COLORS = {
    "Value Iteration": "steelblue",
    "REINFORCE": "tomato",
    "Q-Learning": "seagreen",
    "Model Based": "mediumpurple",
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
    "REINFORCE": "reinforce",
    "Q-Learning": "qlearning",
    "Model Based": "model_based",
}

SEEDS_CALIBRATION = list(config.SEEDS_DEFAULT)
SEEDS_MAIN_SMALL = list(config.SEEDS_DEFAULT)
SEEDS_MAIN_LARGE = list(config.SEEDS_LARGE_MAP)
MAP_SEED = config.SEED


def setup_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


# -- Utilidades de reproducibilidad y config ---------------------------------

def _set_seed(agent_seed, env=None, env_seed=None):
    """
    Fija las semillas del agente y, opcionalmente, del entorno.
    """
    np.random.seed(agent_seed)
    random.seed(agent_seed)

    if env is not None and env_seed is not None:
        env.reset(seed=env_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(env_seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(env_seed)

    return env_seed


@contextlib.contextmanager
def _temporary_config(**values):
    originals = {name: getattr(config, name) for name in values}
    try:
        for name, value in values.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(config, name, value)


def _label_float(value):
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _is_large_map(map_name):
    return map_name not in {"4x4", "8x8"}


def _effective_reward_schedule(reward_schedule):
    return tuple(reward_schedule if reward_schedule is not None else config.REWARD_SCHEDULE)


def _hyperparams_snapshot(gamma):
    return {
        "gamma": gamma,
        "learning_rate": config.LEARNING_RATE,
        "lr_decay": config.LR_DECAY,
        "epsilon": config.EPSILON,
        "epsilon_decay": config.EPSILON_DECAY,
        "epsilon_min": config.EPSILON_MIN,
        "reinforce_lr": config.REINFORCE_LR,
        "num_transitions_mb": config.NUM_TRANSITIONS_MB,
        "planning_steps_per_iter": config.PLANNING_STEPS_PER_ITER,
        "theta_convergence": config.THETA_CONVERGENCE,
        "t_max": config.T_MAX,
    }


# -- CSV ----------------------------------------------------------------------

def _csv_path(prefix, exp_id):
    safe_exp_id = str(exp_id).replace(" ", "_").replace("/", "_")
    return os.path.join(RESULTS_DIR, f"{prefix}_{safe_exp_id}.csv")


def _init_results_csv(exp_id):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _csv_path("results", exp_id)
    if not os.path.exists(path):
        with open(path, "w", newline="") as file:
            csv.writer(file).writerow(RESULTS_CSV_HEADER)
    return path


def _append_results_csv(exp_id, run):
    path = _init_results_csv(exp_id)
    row = [
        exp_id,
        run["algorithm"],
        run["map_name"],
        run["success_rate"],
        str(tuple(run["reward_schedule"])),
        run["agent_seed"],
        run["env_seed"],
        run["map_seed"],
        json.dumps(run["hyperparams"], sort_keys=True),
        run["total_train_time_s"],
        run["n_train_episodes"],
        run["n_train_iterations"],
        run["eval_success_rate"],
        run["eval_mean_reward"],
        run["eval_mean_steps"],
        run["eval_termination_goal_pct"],
        run["eval_termination_hole_pct"],
        run["eval_truncation_pct"],
        run.get("eval_policy_agreement_vi"),
        run.get("n_state_action_visited_pct"),
        run.get("final_q_diff_l_inf"),
    ]
    with open(path, "a", newline="") as file:
        csv.writer(file).writerow(row)


def _init_curves_csv(exp_id):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _csv_path("learning_curves", exp_id)
    if not os.path.exists(path):
        with open(path, "w", newline="") as file:
            csv.writer(file).writerow(CURVES_CSV_HEADER)
    return path


def _append_curves_csv(exp_id, run):
    history = run.get("history")
    if not isinstance(history, list) or not history:
        return
    if run["algorithm"] == "Value Iteration":
        return

    path = _init_curves_csv(exp_id)
    is_episodic = run["algorithm"] in EPISODE_DEPENDENT

    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        for index, value in enumerate(history, start=1):
            writer.writerow([
                exp_id,
                run["algorithm"],
                run["agent_seed"],
                index,
                value,
                None,
                1 if is_episodic and value > 0 else None,
                None,
                None,
                None,
            ])


# -- Entrenamiento y evaluacion ----------------------------------------------

def train_agent(agent):
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = agent.train()
    elapsed = time.time() - start
    history = result if isinstance(result, list) else []
    return result, history, elapsed


def _train_iterations(agent_name, train_result, history):
    if agent_name == "Value Iteration" and isinstance(train_result, int):
        return train_result
    if agent_name == "Model Based":
        return len(history)
    return None


def run_single(agent_name, map_name, success_rate, gamma, num_episodes,
               agent_seed, env_seed=None, reward_schedule=None,
               compute_oracle=True):
    """
    Entrena y evalua un agente con una semilla concreta.
    Devuelve todas las metricas necesarias para los CSVs del plan.
    """
    original_gamma = config.GAMMA
    original_num_episodes = config.NUM_EPISODES
    original_t_max = config.T_MAX
    env = None

    try:
        _set_seed(agent_seed)
        config.GAMMA = gamma
        config.NUM_EPISODES = num_episodes
        config.T_MAX = config.get_t_max(map_name)

        effective_reward = _effective_reward_schedule(reward_schedule)
        map_seed = MAP_SEED if _is_large_map(map_name) else None

        env = create_env(
            map_name=map_name,
            success_rate=success_rate,
            reward_schedule=effective_reward,
            map_seed=MAP_SEED,
        )
        if env_seed is not None:
            _set_seed(agent_seed, env=env, env_seed=env_seed)

        agent = build_agent(AGENT_KEYS[agent_name], env)
        train_result, history, elapsed = train_agent(agent)

        eval_metrics = evaluate_agent(agent, env)
        coverage = compute_state_action_coverage(agent, env)

        oracle_metrics = {
            "eval_policy_agreement_vi": None,
            "final_q_diff_l_inf": None,
        }
        if agent_name == "Value Iteration":
            oracle_metrics["eval_policy_agreement_vi"] = 100.0
        elif compute_oracle and map_name in {"4x4", "8x8"}:
            try:
                oracle_metrics = evaluate_agent_with_oracle(agent, env)
            except Exception as exc:
                print(f"      [oracle skip: {exc}]")

        return {
            "agent": agent_name,
            "algorithm": agent_name,
            "map_name": map_name,
            "success_rate": success_rate,
            "reward_schedule": effective_reward,
            "agent_seed": agent_seed,
            "env_seed": env_seed,
            "map_seed": map_seed,
            "gamma": gamma,
            "num_episodes": num_episodes,
            "hyperparams": _hyperparams_snapshot(gamma),
            "total_train_time_s": elapsed,
            "n_train_episodes": num_episodes,
            "n_train_iterations": _train_iterations(agent_name, train_result, history),
            "n_state_action_visited_pct": coverage,
            "history": history,
            **eval_metrics,
            **oracle_metrics,
        }
    finally:
        if env is not None:
            env.close()
        config.GAMMA = original_gamma
        config.NUM_EPISODES = original_num_episodes
        config.T_MAX = original_t_max


def _aggregate_runs(runs):
    """Calcula media y std de las metricas numericas entre semillas."""
    numeric_keys = [
        "total_train_time_s",
        "n_train_episodes",
        "n_train_iterations",
        "eval_success_rate",
        "eval_mean_reward",
        "eval_mean_steps",
        "eval_termination_goal_pct",
        "eval_termination_hole_pct",
        "eval_truncation_pct",
        "eval_policy_agreement_vi",
        "n_state_action_visited_pct",
        "final_q_diff_l_inf",
    ]
    agg = {}
    for key in numeric_keys:
        vals = [
            run[key] for run in runs
            if run.get(key) is not None and not isinstance(run.get(key), bool)
        ]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        else:
            agg[f"{key}_mean"] = None
            agg[f"{key}_std"] = None

    agg["history"] = runs[0].get("history", []) if runs else []
    agg["runs"] = runs
    return agg


def run_agents(agents_to_run, map_name, success_rate, gamma,
               num_episodes, save, exp_label, seeds=None,
               reward_schedule=None):
    """
    Entrena y evalua cada agente con multiples semillas.
    Guarda una fila por (agente, semilla) y curvas por episodio/iteracion.
    """
    if seeds is None:
        seeds = SEEDS_CALIBRATION

    results_per_agent = {}
    for name in agents_to_run:
        print(f"    [{name}] entrenando ({len(seeds)} semillas)...", flush=True)
        runs = []
        for seed in seeds:
            run = run_single(
                name,
                map_name,
                success_rate,
                gamma,
                num_episodes,
                agent_seed=seed,
                env_seed=seed,
                reward_schedule=reward_schedule,
            )
            runs.append(run)
            if save:
                _append_results_csv(exp_label, run)
                _append_curves_csv(exp_label, run)

        aggregated = _aggregate_runs(runs)
        results_per_agent[name] = aggregated
        print(
            f"      exito={aggregated['eval_success_rate_mean']:.1f}% "
            f"(+/-{aggregated['eval_success_rate_std']:.1f}) | "
            f"reward={aggregated['eval_mean_reward_mean']:.3f} | "
            f"tiempo={aggregated['total_train_time_s_mean']:.2f}s"
        )

    return results_per_agent


def _best_value_by_metric(results_dict, metric="eval_success_rate_mean"):
    """Devuelve la clave cuyo resultado maximiza la metrica indicada."""
    return max(
        results_dict.keys(),
        key=lambda key: (
            -float("inf")
            if results_dict[key].get(metric) is None
            else results_dict[key].get(metric)
        ),
    )


# -- Calibracion A: Gamma -----------------------------------------------------

def calibration_gamma(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION A - Efecto de gamma")
    print("=" * 55)

    gammas = list(config.CAL_A_GAMMAS)
    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE

    all_results = {}
    with _temporary_config(
        LEARNING_RATE=config.CAL_A_BASE_LEARNING_RATE,
        EPSILON=config.CAL_A_BASE_EPSILON,
        EPSILON_DECAY=config.CAL_A_BASE_EPSILON_DECAY,
        EPSILON_MIN=config.CAL_A_BASE_EPSILON_MIN,
        LR_DECAY=config.CAL_A_BASE_LR_DECAY,
        REINFORCE_LR=config.CAL_A_BASE_REINFORCE_LR,
        NUM_TRANSITIONS_MB=config.CAL_A_NUM_TRANSITIONS_MB,
        THETA_CONVERGENCE=config.CAL_A_THETA_CONVERGENCE,
    ):
        for gamma in gammas:
            print(f"\n  gamma={gamma} | mapa={map_name} | SR={sr:.3f}")
            label = f"cal_A_g{_label_float(gamma)}"
            result = {}
            result.update(run_agents(
                ["Value Iteration", "Model Based", "Q-Learning"],
                map_name, sr, gamma, config.CAL_A_Q_EPISODES, save, label,
                seeds=SEEDS_CALIBRATION,
            ))
            result.update(run_agents(
                ["REINFORCE"],
                map_name, sr, gamma, config.CAL_A_REINFORCE_EPISODES, save, label,
                seeds=SEEDS_CALIBRATION,
            ))
            all_results[gamma] = result

    if save:
        _plot_line(all_results, gammas, "Gamma",
                   "Cal. A: exito vs gamma", "calA_gamma.png")
        _plot_time_line(all_results, gammas, "Gamma",
                        "Cal. A: tiempo vs gamma", "calA_gamma_time.png")

    return all_results


# -- Calibracion B: Episodios -------------------------------------------------

def calibration_episodes(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION B - Numero de episodios")
    print("=" * 55)

    episode_counts = list(config.CAL_B_EPISODE_COUNTS)
    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    agents = [agent for agent in ACTIVE_AGENTS if agent in EPISODE_DEPENDENT]

    all_results = {}
    for num_ep in episode_counts:
        print(f"\n  episodios={num_ep} | mapa={map_name} | SR={sr:.3f}")
        label = f"cal_B_ep{num_ep}"
        all_results[num_ep] = run_agents(
            agents, map_name, sr, gamma, num_ep, save, label,
            seeds=SEEDS_CALIBRATION,
        )

    if save:
        _plot_line(all_results, episode_counts, "Episodios",
                   "Cal. B: exito vs episodios", "calB_episodes.png")
        _plot_curves(_histories_from_result(all_results[episode_counts[-1]]),
                     f"Cal. B: curvas {episode_counts[-1]} episodios",
                     f"calB_curves_{episode_counts[-1]}ep.png")

    return all_results


# -- Calibracion C: Reward schedule ------------------------------------------

def calibration_reward(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION C - Senal de recompensa")
    print("=" * 55)

    reward_configs = dict(config.CAL_C_REWARD_CONFIGS)
    agents = [agent for agent in ACTIVE_AGENTS if agent in EPISODE_DEPENDENT]
    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    all_results = {}
    for reward_name, reward_schedule in reward_configs.items():
        print(f"\n  reward={reward_name} | mapa={map_name} | SR={sr:.3f}")
        label = f"cal_C_{reward_name}"
        all_results[reward_name] = run_agents(
            agents, map_name, sr, gamma, num_ep, save, label,
            seeds=SEEDS_CALIBRATION,
            reward_schedule=reward_schedule,
        )

    if save:
        _plot_reward_bars(all_results, list(reward_configs.keys()), agents)
        for reward_name, result in all_results.items():
            _plot_curves(_histories_from_result(result),
                         f"Cal. C: curvas {reward_name}",
                         f"calC_curves_{reward_name}.png")

    return all_results


# -- Calibracion D: Epsilon y alpha Q-Learning -------------------------------

def calibration_epsilon(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION D - Epsilon y alpha (Q-Learning)")
    print("=" * 55)

    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES
    epsilons = list(config.CAL_D_EPSILONS)
    alphas = list(config.CAL_D_ALPHAS)

    results_eps = {}
    histories_eps = {}
    original_epsilon = config.EPSILON
    original_lr = config.LEARNING_RATE

    try:
        print("\n  D1 - Variando epsilon")
        config.LEARNING_RATE = config.CAL_D_ANCHOR_ALPHA
        for epsilon in epsilons:
            config.EPSILON = epsilon
            print(f"\n  epsilon={epsilon} | alpha={config.LEARNING_RATE}")
            label = f"cal_D1_eps{_label_float(epsilon)}"
            result = run_agents(
                ["Q-Learning"], map_name, sr, gamma, num_ep, save, label,
                seeds=SEEDS_CALIBRATION,
            )
            results_eps[epsilon] = result["Q-Learning"]
            histories_eps[epsilon] = result["Q-Learning"]["history"]

        best_eps = _best_value_by_metric(results_eps)
        print(f"\n  Ganador D1: epsilon={best_eps}")

        print("\n  D2 - Variando alpha")
        config.EPSILON = best_eps
        results_alpha = {}
        histories_alpha = {}
        for alpha in alphas:
            config.LEARNING_RATE = alpha
            print(f"\n  epsilon={best_eps} | alpha={alpha}")
            label = f"cal_D2_alpha{_label_float(alpha)}"
            result = run_agents(
                ["Q-Learning"], map_name, sr, gamma, num_ep, save, label,
                seeds=SEEDS_CALIBRATION,
            )
            results_alpha[alpha] = result["Q-Learning"]
            histories_alpha[alpha] = result["Q-Learning"]["history"]
    finally:
        config.EPSILON = original_epsilon
        config.LEARNING_RATE = original_lr

    if save:
        _plot_line_single(results_eps, epsilons, "Epsilon inicial",
                          "Cal. D1: exito vs epsilon",
                          "calD1_epsilon.png", "Q-Learning")
        _plot_line_single(results_alpha, alphas, "Learning rate",
                          "Cal. D2: exito vs alpha",
                          "calD2_alpha.png", "Q-Learning")
        _plot_curves({f"epsilon={best_eps}": histories_eps[best_eps]},
                     f"Cal. D1: curva epsilon={best_eps}",
                     f"calD1_curves_eps{_label_float(best_eps)}.png")
        best_alpha = _best_value_by_metric(results_alpha)
        _plot_curves({f"alpha={best_alpha}": histories_alpha[best_alpha]},
                     f"Cal. D2: curva alpha={best_alpha}",
                     f"calD2_curves_alpha{_label_float(best_alpha)}.png")

    return {"epsilon": results_eps, "alpha": results_alpha}


# -- Calibracion E: Alpha REINFORCE ------------------------------------------

def calibration_reinforce_lr(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION E - Learning rate REINFORCE")
    print("=" * 55)

    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES
    learning_rates = list(config.CAL_E_REINFORCE_LRS)

    original_lr = config.REINFORCE_LR
    results_lr = {}
    histories_lr = {}
    try:
        for lr in learning_rates:
            config.REINFORCE_LR = lr
            print(f"\n  lr={lr} | mapa={map_name} | SR={sr:.3f}")
            label = f"cal_E_lr{_label_float(lr)}"
            result = run_agents(
                ["REINFORCE"], map_name, sr, gamma, num_ep, save, label,
                seeds=SEEDS_CALIBRATION,
            )
            results_lr[lr] = result["REINFORCE"]
            histories_lr[lr] = result["REINFORCE"]["history"]
    finally:
        config.REINFORCE_LR = original_lr

    if save:
        _plot_line_single(results_lr, learning_rates, "Learning rate",
                          "Cal. E: exito vs LR (REINFORCE)",
                          "calE_reinforce_lr.png", "REINFORCE")
        best_lr = _best_value_by_metric(results_lr)
        _plot_curves({f"lr={best_lr}": histories_lr[best_lr]},
                     f"Cal. E: curva LR={best_lr}",
                     f"calE_curves_lr{_label_float(best_lr)}.png")

    return results_lr


# -- Calibracion F: Transiciones Model Based ---------------------------------

def calibration_transitions(save=False):
    print("\n" + "=" * 55)
    print("CALIBRACION F - Transiciones muestreadas (Model Based)")
    print("=" * 55)

    map_name = "4x4"
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES
    transition_counts = list(config.CAL_F_TRANSITION_COUNTS)

    original_transitions = config.NUM_TRANSITIONS_MB
    results_transitions = {}
    try:
        for transitions in transition_counts:
            config.NUM_TRANSITIONS_MB = transitions
            print(f"\n  transiciones={transitions} | mapa={map_name} | SR={sr:.3f}")
            label = f"cal_F_trans{transitions}"
            result = run_agents(
                ["Model Based"], map_name, sr, gamma, num_ep, save, label,
                seeds=SEEDS_CALIBRATION,
            )
            results_transitions[transitions] = result["Model Based"]
    finally:
        config.NUM_TRANSITIONS_MB = original_transitions

    if save:
        _plot_line_single(results_transitions, transition_counts,
                          "Transiciones totales",
                          "Cal. F: exito vs transiciones",
                          "calF_transitions.png", "Model Based")
        _plot_line_single_time(results_transitions, transition_counts,
                               "Transiciones totales",
                               "Cal. F: tiempo vs transiciones",
                               "calF_transitions_time.png", "Model Based")

    return results_transitions


# -- Experimentos principales -------------------------------------------------

def experiment_scaling(save=False, scaled_budget=False):
    label = "presupuesto escalado" if scaled_budget else "presupuesto fijo"
    print("\n" + "=" * 55)
    print(f"EXPERIMENTO PRINCIPAL - Escalado del mapa ({label})")
    print("=" * 55)

    maps = list(config.SCALING_MAPS)
    sr = config.EXPERIMENT_BASE_SUCCESS_RATE
    gamma = config.GAMMA
    base_episodes = config.NUM_EPISODES
    base_transitions = config.NUM_TRANSITIONS_MB
    sub_label = "scaled" if scaled_budget else "fixed"

    all_results = {}
    try:
        for map_name in maps:
            seeds = SEEDS_MAIN_LARGE if _is_large_map(map_name) else SEEDS_MAIN_SMALL
            if scaled_budget:
                size = int(map_name.split("x")[0])
                scale = (size / 4) ** 2
                num_ep = int(base_episodes * scale)
                config.NUM_TRANSITIONS_MB = int(base_transitions * scale)
            else:
                num_ep = base_episodes
                config.NUM_TRANSITIONS_MB = base_transitions

            print(
                f"\n  mapa={map_name} | episodios={num_ep} | "
                f"transiciones_MB={config.NUM_TRANSITIONS_MB} | semillas={len(seeds)}"
            )
            exp_label = f"exp_scaling_{sub_label}_{map_name}"
            all_results[map_name] = run_agents(
                ACTIVE_AGENTS, map_name, sr, gamma, num_ep, save, exp_label,
                seeds=seeds,
            )
    finally:
        config.NUM_TRANSITIONS_MB = base_transitions

    if save:
        _plot_scaling(all_results, maps, sub_label)

    return all_results


def experiment_stochasticity(save=False):
    print("\n" + "=" * 55)
    print("EXPERIMENTO PRINCIPAL - Variacion de success_rate")
    print("=" * 55)

    maps = ["4x4", "8x8"]
    success_rates = list(config.STOCHASTICITY_SUCCESS_RATES)
    gamma = config.GAMMA
    num_ep = config.NUM_EPISODES

    all_results = {map_name: {} for map_name in maps}
    for map_name in maps:
        for sr in success_rates:
            print(f"\n  mapa={map_name} | SR={sr:.3f}")
            exp_label = f"exp_stochasticity_{map_name}_sr{_label_float(sr)}"
            all_results[map_name][sr] = run_agents(
                ACTIVE_AGENTS, map_name, sr, gamma, num_ep, save, exp_label,
                seeds=SEEDS_MAIN_SMALL,
            )

    if save:
        _plot_stochasticity(all_results, maps, success_rates)

    return all_results


# -- Sanity checks ------------------------------------------------------------

def sanity_checks(save=False):
    print("\n" + "=" * 55)
    print("SANITY CHECKS")
    print("=" * 55)

    failures = []

    print("\n  Sanity-1: 4x4 determinista (is_slippery=False)")
    with _temporary_config(
        SLIPPERY=False,
        EPSILON=config.SANITY_EPSILON,
        EPSILON_DECAY=config.SANITY_EPSILON_DECAY,
        REINFORCE_LR=config.SANITY_REINFORCE_LR,
    ):
        for agent_name in ACTIVE_AGENTS:
            seed_results = []
            for seed in SEEDS_MAIN_LARGE:
                run = run_single(
                    agent_name, "4x4", config.SANITY_DETERMINISTIC_SUCCESS_RATE,
                    config.GAMMA, config.SANITY_NUM_EPISODES,
                    agent_seed=seed, env_seed=seed,
                    compute_oracle=False,
                )
                seed_results.append(run["eval_success_rate"])
            mean_success = float(np.mean(seed_results))
            ok = mean_success >= config.SANITY_SUCCESS_THRESHOLD
            print(f"    {'OK' if ok else 'FAIL'} {agent_name}: {mean_success:.1f}%")
            if not ok:
                failures.append(f"Sanity-1 {agent_name}: {mean_success:.1f}%")

    print("\n  Sanity-2: Model Based vs VI policy agreement")
    with _temporary_config(
        NUM_TRANSITIONS_MB=config.SANITY_MB_TRANSITIONS,
        THETA_CONVERGENCE=config.SANITY_POLICY_THETA_CONVERGENCE,
    ):
        run = run_single(
            "Model Based", "4x4", config.EXPERIMENT_BASE_SUCCESS_RATE, config.GAMMA, config.NUM_EPISODES,
            agent_seed=42, env_seed=42,
            compute_oracle=True,
        )
    agreement = run.get("eval_policy_agreement_vi")
    ok = agreement is not None and agreement >= config.SANITY_POLICY_AGREEMENT_THRESHOLD
    print(f"    {'OK' if ok else 'FAIL'} agreement={agreement}")
    if not ok:
        failures.append(f"Sanity-2 policy agreement: {agreement}")

    print("\n  Sanity-3: Value Iteration convergence")
    with _temporary_config(THETA_CONVERGENCE=config.SANITY_VI_THETA_CONVERGENCE):
        run = run_single(
            "Value Iteration", "4x4", config.EXPERIMENT_BASE_SUCCESS_RATE, config.GAMMA, config.NUM_EPISODES,
            agent_seed=42, env_seed=42,
            compute_oracle=False,
        )
    iterations = run.get("n_train_iterations")
    ok = iterations is not None and iterations <= config.SANITY_VI_MAX_ITERATIONS
    print(f"    {'OK' if ok else 'FAIL'} iterations={iterations}")
    if not ok:
        failures.append(f"Sanity-3 VI iterations: {iterations}")

    if save:
        exp_label = "sanity"
        for failure in failures:
            print(f"    fallo registrado: {failure}")

    if failures:
        print("\nFallos de sanity detectados:")
        for failure in failures:
            print(f"    - {failure}")
        return False

    print("\nTodos los sanity checks pasan.")
    return True


# -- Graficas -----------------------------------------------------------------

def _histories_from_result(result_by_agent):
    return {
        agent: stats.get("history", [])
        for agent, stats in result_by_agent.items()
        if stats.get("history")
    }


def _plot_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["eval_success_rate_mean"] for x in x_values]
        stds = [all_results[x][agent]["eval_success_rate_std"] or 0 for x in x_values]
        color = COLORS.get(agent, "gray")
        ax.plot(x_values, vals, marker="o", label=agent, color=color)
        ax.fill_between(x_values,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.15, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="% Exito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_time_line(all_results, x_values, xlabel, title, filename):
    agent_names = list(all_results[x_values[0]].keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[x][agent]["total_train_time_s_mean"] for x in x_values]
        color = COLORS.get(agent, "gray")
        ax.plot(x_values, vals, marker="s", label=agent, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_line_single(results, x_values, xlabel, title, filename, agent_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[x]["eval_success_rate_mean"] for x in x_values]
    stds = [results[x]["eval_success_rate_std"] or 0 for x in x_values]
    color = COLORS.get(agent_name, "gray")
    ax.plot(x_values, vals, marker="o", color=color, label=agent_name)
    ax.fill_between(x_values,
                    [v - s for v, s in zip(vals, stds)],
                    [v + s for v, s in zip(vals, stds)],
                    alpha=0.15, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel="% Exito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_line_single_time(results, x_values, xlabel, title, filename, agent_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[x]["total_train_time_s_mean"] for x in x_values]
    color = COLORS.get(agent_name, "gray")
    ax.plot(x_values, vals, marker="s", color=color, label=agent_name)
    ax.set(title=title, xlabel=xlabel, ylabel="Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_curves(histories_by_agent, title, filename):
    if not histories_by_agent:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, history in histories_by_agent.items():
        color = COLORS.get(name, "gray")
        ax.plot(history, alpha=0.2, color=color)
        smooth = (
            np.convolve(history, np.ones(100) / 100, mode="valid")
            if len(history) >= 100 else history
        )
        ax.plot(smooth, label=name, color=color)
    ax.set(title=title, xlabel="Episodio / iteracion", ylabel="Recompensa / diff")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


def _plot_reward_bars(all_results, reward_labels, agents):
    fig, axes = plt.subplots(1, len(agents), figsize=(6 * len(agents), 5))
    if len(agents) == 1:
        axes = [axes]
    for ax, agent in zip(axes, agents):
        vals = [
            all_results[label][agent]["eval_success_rate_mean"]
            for label in reward_labels
        ]
        bars = ax.bar(reward_labels, vals, color=["steelblue", "tomato", "seagreen"])
        ax.set_title(agent)
        ax.set_ylabel("% Exito")
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=9)
        ax.grid(axis="y")
    fig.suptitle("Cal. C: exito por reward_schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "calC_reward.png"))
    plt.close()


def _plot_scaling(all_results, maps, sub_label):
    x = np.arange(len(maps))
    agent_names = list(all_results[maps[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[map_name][agent]["eval_success_rate_mean"] for map_name in maps]
        stds = [all_results[map_name][agent]["eval_success_rate_std"] or 0 for map_name in maps]
        color = COLORS.get(agent, "gray")
        ax.plot(x, vals, marker="o", label=agent, color=color)
        ax.fill_between(x,
                        [v - s for v, s in zip(vals, stds)],
                        [v + s for v, s in zip(vals, stds)],
                        alpha=0.15, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.set(title=f"Escalado ({sub_label}): exito vs mapa",
           xlabel="Mapa", ylabel="% Exito", ylim=(0, 110))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"exp_scaling_{sub_label}_success.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent in agent_names:
        vals = [all_results[map_name][agent]["total_train_time_s_mean"] for map_name in maps]
        ax.plot(x, vals, marker="s", label=agent, color=COLORS.get(agent, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.set(title=f"Escalado ({sub_label}): tiempo vs mapa",
           xlabel="Mapa", ylabel="Tiempo (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"exp_scaling_{sub_label}_time.png"))
    plt.close()


def _plot_stochasticity(all_results, maps, success_rates):
    for map_name in maps:
        fig, ax = plt.subplots(figsize=(10, 5))
        agent_names = list(all_results[map_name][success_rates[0]].keys())
        for agent in agent_names:
            vals = [
                all_results[map_name][sr][agent]["eval_success_rate_mean"]
                for sr in success_rates
            ]
            stds = [
                all_results[map_name][sr][agent]["eval_success_rate_std"] or 0
                for sr in success_rates
            ]
            color = COLORS.get(agent, "gray")
            ax.plot(success_rates, vals, marker="o", label=agent, color=color)
            ax.fill_between(success_rates,
                            [v - s for v, s in zip(vals, stds)],
                            [v + s for v, s in zip(vals, stds)],
                            alpha=0.15, color=color)
        ax.set(title=f"Estocasticidad {map_name}: exito vs success_rate",
               xlabel="success_rate", ylabel="% Exito", ylim=(0, 110))
        ax.set_xticks(success_rates)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"exp_stochasticity_{map_name}.png"))
        plt.close()


# -- CLI ----------------------------------------------------------------------

def _build_runners():
    return {
        "sanity": sanity_checks,
        "calibration_gamma": calibration_gamma,
        "calibration_episodes": calibration_episodes,
        "calibration_reward": calibration_reward,
        "calibration_epsilon": calibration_epsilon,
        "calibration_reinforce_lr": calibration_reinforce_lr,
        "calibration_transitions": calibration_transitions,
        "exp_scaling_fixed": lambda save: experiment_scaling(save, scaled_budget=False),
        "exp_scaling_scaled": lambda save: experiment_scaling(save, scaled_budget=True),
        "exp_stochasticity": experiment_stochasticity,
    }


if __name__ == "__main__":
    runners = _build_runners()
    parser = argparse.ArgumentParser(description="Experimentos FrozenLake")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", *runners.keys()],
        help="Experimento a ejecutar. 'all' ejecuta todos en orden.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guarda CSVs y graficas en results/",
    )
    args = parser.parse_args()

    setup_dirs()

    if args.exp == "all":
        for runner in runners.values():
            runner(save=args.save)
    else:
        runners[args.exp](save=args.save)

    if args.save:
        print(f"\nResultados generados en '{RESULTS_DIR}/'.")
