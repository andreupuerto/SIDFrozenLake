<!--
README.md
Changes: rewrote execution, repository structure, parametrization, custom-map,
success_rate, and CSV-output documentation to match the required README
structure from Section 5.
-->

# FrozenLake RL - Practica 2 SID

Repositorio para comparar cuatro algoritmos de aprendizaje por refuerzo en
`FrozenLake-v1` de Gymnasium:

- Value Iteration
- Direct Estimation / Model-Based Monte Carlo
- Q-Learning
- REINFORCE

El flujo experimental usa siempre ejecucion por CLI, multiples semillas y dos
CSV por experimento: resultados finales y curvas de aprendizaje.

## Instalacion

```bash
pip install "gymnasium>=1.3" numpy matplotlib pygame
```

## Estructura

- `config.py`: hiperparametros, semillas, presupuestos y configuracion base.
- `main.py`: creacion del entorno, evaluacion, diagnosticos y constructor de agentes.
- `value_iteration.py`: agente Value Iteration.
- `model_based.py`: agente Direct Estimation / Model-Based Monte Carlo.
- `qlearning.py`: agente Q-Learning.
- `reinforce.py`: agente REINFORCE.
- `experiments.py`: sanity checks, calibraciones y experimentos principales.
- `results/`: CSVs y graficas generadas con `--save`.

## Ejecutar Experimentos

### Sanity checks

```bash
python experiments.py --exp sanity
```

### Fase de calibracion

```bash
python experiments.py --exp calibration_gamma --save
python experiments.py --exp calibration_episodes --save
python experiments.py --exp calibration_reward --save
python experiments.py --exp calibration_epsilon --save
python experiments.py --exp calibration_reinforce_lr --save
python experiments.py --exp calibration_transitions --save
```

### Experimentos principales

```bash
python experiments.py --exp exp_scaling_fixed --save
python experiments.py --exp exp_scaling_scaled --save
python experiments.py --exp exp_stochasticity --save
```

### Todo en orden

```bash
python experiments.py --exp all --save
```

## Personalizacion

Modifica `config.py` para cambiar valores por defecto:

- `MAP_NAME`: `"4x4"`, `"8x8"` o mapas custom como `"10x10"` y `"12x12"`.
- `SUCCESS_RATE`: probabilidad de ejecutar la accion deseada.
- `EXPERIMENT_BASE_SUCCESS_RATE`: valor fijo usado por calibraciones y escalado.
- `REWARD_SCHEDULE`: tupla `(goal, hole, frozen)` pasada a Gymnasium.
- `NUM_EPISODES`: presupuesto base para Q-Learning y REINFORCE.
- `NUM_TRANSITIONS_MB`: presupuesto total para Model-Based.
- `SEEDS_DEFAULT` y `SEEDS_LARGE_MAP`: semillas usadas en experimentacion.

## Mapas Custom

Los mapas `NxN` se generan con `generate_random_map(size=N, seed=config.SEED)`.
Para mapas custom, `main.create_env()` envuelve el entorno con `TimeLimit` y usa
un horizonte `4*N*N`, definido por `config.get_t_max(map_name)`.

Ejemplo:

```python
import main

env = main.create_env(map_name="12x12", success_rate=1/3, map_seed=99)
```

## Success Rate

`success_rate` controla la probabilidad de que se ejecute la accion elegida en
modo `is_slippery=True`. Los experimentos principales prueban:

```python
[1.0, 0.8, 0.6, 1/3, 0.2]
```

El valor base del plan es `1/3`, equivalente al comportamiento estocastico por
defecto.

## Formato de Salida

Con `--save`, cada configuracion experimental genera:

- `results/results_<exp_id>.csv`
- `results/learning_curves_<exp_id>.csv`

### `results_<exp_id>.csv`

Una fila por `(algoritmo, semilla)`:

```text
exp_id, algorithm, map_name, success_rate, reward_schedule,
agent_seed, env_seed, map_seed, hyperparams_json,
total_train_time_s, n_train_episodes, n_train_iterations,
eval_success_rate, eval_mean_reward, eval_mean_steps,
eval_termination_goal_pct, eval_termination_hole_pct, eval_truncation_pct,
eval_policy_agreement_vi, n_state_action_visited_pct, final_q_diff_l_inf
```

### `learning_curves_<exp_id>.csv`

Una fila por `(algoritmo, semilla, episodio/iteracion)` cuando el agente devuelve
historial de entrenamiento:

```text
exp_id, algorithm, agent_seed, episode,
train_reward, train_steps, train_success, epsilon, alpha, elapsed_time_s
```

Las columnas no capturadas por los agentes actuales se dejan vacias.
