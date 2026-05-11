# Plan de Experimentación Final — SID Práctica 2

**Entorno:** FrozenLake-v1 (Gymnasium, slippery)
**Algoritmos:** Value Iteration, Direct Estimation, Q-Learning, REINFORCE
**Versión:** v1.0 — consolidada
**Fecha objetivo de entrega:** 11/05/2026

---

## 1. Configuración General de la Experimentación

### 1.1 Parámetros invariantes del entorno

Estos valores **no se modifican nunca** salvo donde un experimento concreto los varíe explícitamente:

| Parámetro | Valor por defecto | Notas |
|---|---|---|
| `is_slippery` | `True` | **Constraint duro del enunciado.** No se desactiva nunca |
| `success_rate` | `1/3` | Default de Gymnasium. Solo se varía en el experimento de estocasticidad |
| `reward_schedule` | `(1, 0, 0)` | Solo se varía en el experimento de reward shaping |
| `map_name` | `"4x4"` | Solo se varía en el experimento de escalado |
| `render_mode` | `None` | Sin renderizado para acelerar experimentos |

### 1.2 Reproducibilidad y semillas

**Esquema de semillas dual:** se separa explícitamente la semilla del agente de la semilla del entorno.

- **Semilla del agente** (`agent_seed`): controla `numpy.random`, `random`, y la inicialización del agente (Q-table, θ de REINFORCE).
- **Semilla del entorno** (`env_seed`): se pasa a `env.reset(seed=env_seed)` y se incrementa por episodio para asegurar diversidad estocástica controlada.
- **Semilla del mapa** (`map_seed`): solo aplicable cuando se usa `generate_random_map`. Se fija y reporta explícitamente en el informe.

**Número de semillas por configuración:**

| Tipo de experimento | Semillas | N_eval | Justificación |
|---|---|---|---|
| Calibración | 5 | 1000 | Entorno fijo 4x4, coste bajo. Aquí es donde más se necesita rigor estadístico |
| Experimento principal 4x4 y 8x8 | 5 | 1000 | Resultados centrales del informe |
| Mapas custom ≥10x10 | 3 | 1000 | El cómputo crece y 3 semillas basta para tendencias de escalabilidad |
| Sanity checks (no-slippery) | 3 | 500 | Solo verificación de correctitud |

### 1.3 Evaluación

**Separación estricta entre entrenamiento y evaluación.** La evaluación se realiza **siempre post-hoc** con:

- Política determinista (greedy):
  - VI/DE: `argmax_a Q(s,a)` derivado del V/Q aprendido.
  - Q-Learning: `argmax_a Q(s,a)` con `epsilon=0`.
  - REINFORCE: `argmax_a π(a|s; θ)` (modo determinista, no muestreo).
- N_eval = 1000 episodios independientes.
- Horizonte de evaluación: usar el `TimeLimit` nativo del entorno (100 para 4x4, 200 para 8x8) o `4·N²` para mapas custom.

### 1.4 Horizontes temporales

- **4x4:** 100 pasos (truncation nativa de Gymnasium).
- **8x8:** 200 pasos (truncation nativa).
- **Mapas custom NxN ≥ 10:** envolver con `gymnasium.wrappers.TimeLimit(max_episode_steps=4*N*N)`.

### 1.5 Formato de salida de los resultados

Todos los experimentos generan **dos ficheros CSV** por run:

#### CSV de resultados agregados (`results_<exp_id>.csv`)
Una fila por (algoritmo × configuración × semilla). Columnas:

```
exp_id, algorithm, map_size, success_rate, reward_schedule,
agent_seed, env_seed, map_seed,
hyperparams_json,
total_train_time_s, n_train_episodes, n_train_iterations,
eval_success_rate, eval_mean_reward, eval_mean_steps,
eval_termination_goal_pct, eval_termination_hole_pct, eval_truncation_pct,
eval_policy_agreement_vi,
n_state_action_visited_pct,
final_q_diff_l_inf
```

#### CSV de curvas de aprendizaje (`learning_curves_<exp_id>.csv`)
Una fila por (algoritmo × configuración × semilla × episodio). Solo para Q-Learning, REINFORCE y DE. Columnas:

```
exp_id, algorithm, agent_seed, episode,
train_reward, train_steps, train_success,
epsilon, alpha,
elapsed_time_s
```

**Convención de nombres de experimentos:** `cal_A`, `cal_B`, `cal_C`, `cal_D1`, `cal_D2`, `cal_E`, `cal_F`, `exp_size`, `exp_stochasticity`, `exp_reward`, `sanity`.

### 1.6 Salida estándar de cada experimento

Cada experimento ejecutable debe imprimir por stdout:

- Línea inicial con configuración completa.
- Una línea por configuración terminada con tiempo y métricas resumen.
- Línea final con localización de los CSV generados.

---

## 2. Sanity Checks (pre-experimentación)

**Propósito:** verificar que las implementaciones son correctas antes de invertir tiempo en experimentos. **Esto se ejecuta una sola vez al inicio**.

| Test | Algoritmos | Configuración | Criterio de éxito |
|---|---|---|---|
| Sanity-1: determinista | Todos | 4x4, `is_slippery=False` | Todos alcanzan ≥99% éxito en evaluación |
| Sanity-2: policy match | DE vs VI | 4x4, slippery, 50000 transiciones DE | Policy agreement ≥90% |
| Sanity-3: convergencia VI | VI | 4x4, slippery, θ=1e-8 | Converge en ≤500 iteraciones |

Si alguno falla, hay un bug y se detiene el proceso experimental.

---

## 3. Fase de Calibración

**Propósito general:** justificar los valores de hiperparámetros que se usarán en los experimentos principales (Fases 4 y 5).

**Configuración base de calibración:** 4x4, `is_slippery=True`, `success_rate=1/3`, `reward_schedule=(1,0,0)`. **No se varían estos parámetros durante la calibración** — el objetivo es aislar el efecto del hiperparámetro estudiado.

### 3.1 Cal A — Factor de descuento γ

**Propósito:** justificar el γ usado en el experimento principal. Compartido entre los cuatro algoritmos.

**Algoritmos:** todos (VI, DE, Q-Learning, REINFORCE).

**Parámetros a probar:**
- γ ∈ {0.90, 0.95, 0.99}

**Parámetros fijos:**
- VI: θ=1e-6
- DE: 5000 transiciones totales, θ=1e-6 para la iteración interna
- Q-Learning: α=0.1, ε₀=1.0, ε_decay=0.999, ε_min=0.01, episodios=10000
- REINFORCE: α=0.01, episodios=20000, sin baseline

**Métricas capturadas:**
- `eval_success_rate`, `eval_mean_reward`, `total_train_time_s`
- Por VI y DE: `n_iterations` hasta convergencia

**Salida esperada:** mejor γ por algoritmo. Hipótesis: γ alto (0.99) gana para VI/DE (recompensa sparse, propagación lejana); para Q-Learning y REINFORCE puede haber matices por la varianza inducida.

### 3.2 Cal B — Número de episodios

**Propósito:** determinar el presupuesto mínimo de episodios para convergencia. Justifica el presupuesto del experimento principal.

**Algoritmos:** Q-Learning, REINFORCE (los únicos que aprenden por episodios).

**Parámetros a probar:**
- Episodios ∈ {500, 1000, 2000, 5000, 10000, 20000}

**Parámetros fijos:**
- γ = ganador de Cal A
- Q-Learning: α=0.1, ε₀=1.0, ε_decay=0.999, ε_min=0.01
- REINFORCE: α=0.01

**Definición operativa de "convergencia":** primer punto donde la media móvil de `train_success` con ventana=200 supera el 70% del valor final asintótico y no cae por debajo en los siguientes 500 episodios.

**Métricas capturadas:**
- Mismas que Cal A.
- Adicionalmente: `episode_to_convergence` calculado post-hoc.

**Salida esperada:** mínimo de episodios para Q-Learning (probablemente 5000-10000) y REINFORCE (probablemente 10000-20000).

### 3.3 Cal C — Señal de recompensa

**Propósito:** comparar el reward shaping con la recompensa por defecto, como pide la nota al pie 2 del enunciado.

**Algoritmos:** Q-Learning, REINFORCE.

**Justificación de la limitación:** VI y DE leen el MDP directamente — DE construye `R̂` a partir de las recompensas que observa, así que cambiar `reward_schedule` también les afecta, pero el análisis es trivial (la política óptima del nuevo MDP es la nueva política óptima). Para Q-Learning y REINFORCE, el reward shaping cambia la dinámica de aprendizaje, que es lo interesante.

**Parámetros a probar (3 configuraciones de `reward_schedule`):**
- `default`: `(1, 0, 0)` — referencia obligatoria por el enunciado
- `hole_penalty`: `(1, -1, 0)` — penalización por caer en agujero
- `step_penalty`: `(1, 0, -0.01)` — penalización por paso

**Parámetros fijos:** γ ganador de Cal A, episodios ganador de Cal B, resto en valores base.

**Métricas capturadas:** todas las primarias. Especial atención a `eval_mean_steps` (esperable que `step_penalty` reduzca pasos).

**Decisión metodológica:** los resultados de Cal C **no se propagan a los experimentos principales**. El experimento principal usa siempre `reward_schedule=(1,0,0)` (default). Cal C es un análisis paralelo cuya finalidad es discutir el efecto del reward shaping en el informe.

### 3.4 Cal D1 — Coeficiente de exploración ε

**Propósito:** justificar el ε₀ usado en Q-Learning para el experimento principal.

**Algoritmos:** Q-Learning únicamente.

**Parámetros a probar:**
- ε₀ ∈ {0.3, 0.5, 0.8, 1.0}

**Parámetros fijos:**
- γ = ganador de Cal A
- α = 0.1 (anchor conservador — se afinará en Cal D2)
- ε_decay = 0.999, ε_min = 0.01
- Episodios = ganador de Cal B

**Métricas:** primarias + curva de aprendizaje (para ver la fase de transición exploración→explotación).

### 3.5 Cal D2 — Tasa de aprendizaje α

**Propósito:** justificar el α usado en Q-Learning para el experimento principal.

**Algoritmos:** Q-Learning únicamente.

**Parámetros a probar:**
- α ∈ {0.05, 0.1, 0.3, 0.5}

**Parámetros fijos:**
- γ = ganador de Cal A
- ε₀ = ganador de Cal D1
- ε_decay = 0.999, ε_min = 0.01
- Episodios = ganador de Cal B

**Métricas:** primarias + varianza inter-semilla del `eval_success_rate` (diagnostica si α causa oscilaciones).

### 3.6 Cal E — Tasa de aprendizaje α de REINFORCE

**Propósito:** justificar el α de REINFORCE. **Imprescindible** porque es el parámetro más sensible del método.

**Algoritmos:** REINFORCE únicamente.

**Parámetros a probar:**
- α ∈ {0.001, 0.01, 0.05, 0.1}

**Parámetros fijos:**
- γ = ganador de Cal A
- Episodios = ganador de Cal B
- Sin baseline (decisión de implementación fija)

**Métricas:** primarias + varianza inter-semilla.

### 3.7 Cal F — Transiciones muestreadas en Direct Estimation

**Propósito:** calibrar el único hiperparámetro de presupuesto de DE.

**Algoritmos:** Direct Estimation únicamente.

**Parámetros a probar:**
- Transiciones totales ∈ {1000, 5000, 20000, 100000}

**Parámetros fijos:**
- γ = ganador de Cal A
- Política de muestreo: uniforme aleatoria (la del notebook base)
- θ = 1e-6 para la iteración interna sobre el modelo estimado

**Métricas:** primarias + `n_state_action_visited_pct` (diagnóstico de cobertura).

### 3.8 Decisiones de implementación fijas (no se calibran)

Para no inflar la batería de experimentos, los siguientes parámetros se fijan por justificación teórica en el informe (sección de Implementación):

| Parámetro | Valor fijo | Justificación |
|---|---|---|
| ε_decay (Q-Learning) | 0.999 | Decay geométrico clásico, suficientemente lento para satisfacer Robbins-Monro en horizonte largo |
| ε_min (Q-Learning) | 0.01 | Mantener algo de exploración residual sin destruir la política |
| α_decay (Q-Learning) | sin decay | Decisión: tratar α como constante moderada y documentar en el informe la tensión con Robbins-Monro |
| Baseline en REINFORCE | desactivado | Mantener el algoritmo "puro" tal como en las slides p.38 |
| Política de muestreo en DE | uniforme aleatoria | Coherente con el notebook base; analizar la limitación de cobertura en el informe |
| θ de convergencia (VI, DE) | 1e-6 | Sweet spot entre precisión y tiempo |

---

## 4. Experimento Principal de Escalado del Mapa (Fase 4)

**Propósito:** estudiar cómo cada algoritmo escala con el tamaño del mapa, manteniendo la estocasticidad fija.

### 4.1 Configuración

**Parámetros que varían:**
- `map_size` / `map_name`:
  - `"4x4"` — 16 estados
  - `"8x8"` — 64 estados
  - `generate_random_map(size=10, seed=42)` — ~100 estados
  - `generate_random_map(size=12, seed=42)` — ~144 estados

**Parámetros fijos:**
- `is_slippery=True`, `success_rate=1/3`
- `reward_schedule=(1, 0, 0)`
- Hiperparámetros de cada algoritmo: **los ganadores de la fase de calibración**

### 4.2 Sub-experimentos

**4.A — Presupuesto fijo:** todos los algoritmos usan el mismo presupuesto de episodios/transiciones que se calibró para 4x4, sin escalarlo. Esto muestra **cómo se degrada el rendimiento con el mismo cómputo** al crecer el mapa.

**4.B — Presupuesto escalado:** el presupuesto se escala proporcionalmente a `|S|·|A|`. Para Q-Learning y REINFORCE: `n_episodios_base × (N/4)²`. Para DE: `n_transiciones_base × (N/4)²`. Esto muestra si las diferencias son **solo cuestión de muestras o degradación intrínseca**.

### 4.3 Algoritmos

Los cuatro algoritmos en todas las configuraciones. Para mapas ≥10x10 se reduce a 3 semillas.

### 4.4 Métricas

Todas las primarias. Adicionalmente:
- **`eval_policy_agreement_vi`** — fracción de estados donde la política aprendida coincide con la de VI. VI sirve de oráculo de la política óptima para mapas donde VI converge.
- **`final_q_diff_l_inf`** — solo para Q-Learning, distancia L∞ entre la Q aprendida y la Q∗ calculada por VI.

---

## 5. Experimento Principal de Estocasticidad (Fase 5)

**Propósito:** estudiar cómo afecta `success_rate` al rendimiento de cada algoritmo, **manteniendo el tamaño del mapa fijo**.

**Justificación de la separación frente al experimento de escalado:** mantener las fases separadas permite atribuir causalidad limpia. Si las uniéramos en un diseño factorial, no podríamos distinguir si la degradación viene del tamaño o de la estocasticidad.

### 5.1 Configuración

**Parámetros que varían:**
- `success_rate` ∈ {1.0, 0.8, 0.6, 1/3 (≈0.333), 0.2}

**Parámetros fijos:**
- `is_slippery=True` (siempre — flag literal del enunciado, aunque con `success_rate=1.0` la dinámica sea determinista efectiva)
- `map_name` ∈ {`"4x4"`, `"8x8"`} — dos cortes para ver si la sensibilidad a la estocasticidad cambia con el tamaño
- `reward_schedule=(1, 0, 0)`
- Hiperparámetros: ganadores de la fase de calibración

### 5.2 Algoritmos

Los cuatro algoritmos en todas las configuraciones. 5 semillas en 4x4, 5 semillas en 8x8.

### 5.3 Métricas

Todas las primarias. Atención especial a:
- **Curvas de degradación** `eval_success_rate` vs `success_rate` por algoritmo.
- **Tiempo a convergencia** vs `success_rate` — hipótesis: crece exponencialmente al bajar success_rate.

### 5.4 Punto teórico de interés

Incluir `success_rate=0.2` permite observar el régimen donde **la acción deseada es menos probable que las laterales combinadas**. La política óptima cambia cualitativamente — esto es un resultado teórico interesante para discutir en el informe.

---

## 6. Resumen de Métricas y Su Justificación

### 6.1 Métricas primarias (todas las pide el enunciado)

| Métrica | Definición operativa | Para qué hipótesis sirve |
|---|---|---|
| `eval_success_rate` | Fracción de N_eval episodios que terminan en goal | Rendimiento global, hipótesis principales |
| `eval_mean_reward` | Recompensa media en evaluación | Más granular que success_rate (lección del informe anterior) |
| `eval_mean_steps` | Pasos medios por episodio en evaluación | Eficiencia de la política, reward shaping |
| `total_train_time_s` | Wall-clock del entrenamiento | Escalabilidad, frontera Pareto |
| `time_per_episode_s` | Tiempo total / nº unidades de aprendizaje | Comparación entre algoritmos |
| `episode_to_convergence` | Primer episodio donde la media móvil supera el 70% del valor asintótico | Velocidad de aprendizaje |

### 6.2 Métricas secundarias (diagnóstico y análisis)

| Métrica | Para qué sirve |
|---|---|
| `eval_termination_goal_pct` / `_hole_pct` / `_truncation_pct` | Distinguir las 3 formas de terminar el episodio. Crítico — lección del informe anterior donde colapsar todo en "fracaso" ocultó información |
| `n_state_action_visited_pct` | Diagnóstico de cobertura — relevante para DE (H2) y Q-Learning (H3) |
| `eval_policy_agreement_vi` | Métrica de "cuán cerca del óptimo" — compara Q-Learning, DE, REINFORCE contra VI |
| `final_q_diff_l_inf` | Para Q-Learning: distancia L∞ a la Q∗ de VI |
| Varianza inter-semilla de `eval_success_rate` | Estabilidad del algoritmo |

### 6.3 Importante — lección del informe del año pasado

El informe del año pasado **midió solo "alcanzar el goal" como métrica de éxito** y obtuvo tasas casi nulas en el modo estocástico para todos los algoritmos. Esto debilitó significativamente sus conclusiones, dejando hipótesis "no concluyentes".

**Mitigación:** capturar siempre `eval_mean_reward` además de `eval_success_rate`. La recompensa media discrimina mucho mejor entre algoritmos que la tasa de éxito binaria cuando esta es uniformemente baja.

---

## 7. Hipótesis Falsables del Estudio

Resumidas por sección del informe a la que corresponden.

| ID | Hipótesis | Validación |
|---|---|---|
| H1 | VI alcanza el mejor rendimiento (techo de optimalidad) en todas las configuraciones donde converge | `eval_success_rate` y `eval_mean_reward` máximos para VI |
| H2 | DE converge a VI pero su rendimiento está acotado por la cobertura (s,a) de su muestreo aleatorio | Correlación entre `n_state_action_visited_pct` y `eval_policy_agreement_vi` |
| H3 | Q-Learning escala peor que VI con el tamaño del mapa, pero converge a soluciones casi óptimas con presupuesto suficiente | Brecha Q-L vs VI crece con N en 4.A pero se cierra en 4.B |
| H4 | REINFORCE tiene mayor varianza y peor sample efficiency que Q-Learning | Varianza inter-semilla mayor; más episodios para convergencia |
| H5 | La estocasticidad degrada asimétricamente los algoritmos | Pendiente de degradación más pronunciada en REINFORCE > Q-L > DE > VI |
| H6 | El reward shaping mejora la velocidad de convergencia pero puede no mejorar el rendimiento final | Cal C — menor `episode_to_convergence` con shaping, similar `eval_success_rate` final |
| H7 | Existe una frontera Pareto tiempo-cómputo vs rendimiento donde el algoritmo dominante cambia con la escala | Visualización Pareto en discusión final |

---

## 8. Blueprint del Informe (Mapeo a la Rúbrica)

| Sección | Mapeo a rúbrica | Páginas estimadas |
|---|---|---|
| 1. Introducción y taxonomía RL | — | 1-2 |
| 2. Decisiones de implementación | **Implementación (20%)** | 2-3 |
| 3. Diseño experimental e hipótesis | **Diseño experimental (20%)** | 3 |
| 4. Resultados — Calibración | **Análisis (20%)** | 2-3 |
| 5. Resultados — Escalado | **Análisis (20%)** | 2-3 |
| 6. Resultados — Estocasticidad | **Análisis (20%)** | 2 |
| 7. Discusión crítica y escalabilidad | **Análisis (20%)** | 2 |
| 8. Conclusiones y recomendaciones | — | 1 |
| README + código modular | **Ejecución (40%)** | (no en informe) |

**Bonus +2:** REINFORCE tratado en pie de igualdad con los otros tres en todas las fases.

---

## 9. Plan de Ejecución Sugerido

Orden de ejecución para optimizar el tiempo total de cómputo:

1. **Sanity checks** (minutos)
2. **Cal A** (γ para los 4 algoritmos) — los demás dependen de su resultado
3. **Cal B** (episodios) — Q-Learning y REINFORCE
4. **Cal D1 → Cal D2** (ε y α de Q-Learning, secuencial)
5. **Cal E** (α de REINFORCE) — en paralelo con D1/D2
6. **Cal F** (transiciones de DE) — en paralelo con cualquier otra
7. **Cal C** (reward shaping) — puede hacerse al final, análisis paralelo
8. **Experimento principal de escalado (4.A y 4.B)** — el más costoso
9. **Experimento principal de estocasticidad** — el más costoso después del anterior

**Estimación de runs totales:**

| Experimento | Configuraciones | Semillas | Runs |
|---|---|---|---|
| Sanity | 5 | 3 | 15 |
| Cal A | 3 γ × 4 algoritmos | 5 | 60 |
| Cal B | 6 episodios × 2 algos | 5 | 60 |
| Cal C | 3 rewards × 2 algos | 5 | 30 |
| Cal D1 | 4 ε | 5 | 20 |
| Cal D2 | 4 α | 5 | 20 |
| Cal E | 4 α | 5 | 20 |
| Cal F | 4 transiciones | 5 | 20 |
| Exp escalado 4.A | 4 mapas × 4 algos | 5 (3 en ≥10x10) | ~60 |
| Exp escalado 4.B | 4 mapas × 4 algos | 5 (3 en ≥10x10) | ~60 |
| Exp estocasticidad | 5 success × 2 mapas × 4 algos | 5 | 200 |
| **TOTAL** | | | **~565 runs** |

Manejable en una máquina de desarrollo razonable si los runs individuales no son muy largos (mapas pequeños). Para mapas ≥10x10 el tiempo puede crecer notablemente, por eso se reduce a 3 semillas.

---

## 10. Decisiones Finales Tomadas Tras el Análisis del Informe del Año Anterior

Resumen de las decisiones de diseño consolidadas en este plan:

1. **Calibración por hiperparámetro (no por algoritmo)** — facilita comparación cruzada en el informe.
2. **Cal A, B, C, D1, D2, E, F como experimentos imprescindibles.** Cal D3 (decay) se trata como decisión de implementación fija.
3. **Fases de escalado y estocasticidad separadas** — preserva atribución de causalidad.
4. **5 semillas estándar, 3 para mapas ≥10x10** — equilibrio entre rigor y coste.
5. **Separación estricta entrenamiento/evaluación** con N_eval=1000 — corrección del defecto metodológico del notebook base.
6. **Métricas múltiples (success_rate + mean_reward + termination breakdown)** — corrección de la limitación del informe del año pasado donde solo medir success_rate llevó a conclusiones débiles.
7. **`success_rate` se varía nativamente vía `gym.make()`** — disponible desde Gymnasium v1.3, no requiere wrapper.
8. **`reward_schedule` se varía nativamente** — no requiere wrapper custom.
9. **Mapas custom con `generate_random_map(seed=...)` y semilla del mapa fija** — reproducibilidad explícita.
10. **Logging granular en CSVs separados** (resultados agregados vs curvas) — facilita el análisis posterior.
