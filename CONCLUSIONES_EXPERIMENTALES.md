# Conclusiones Experimentales — FrozenLake-v1

Estudio del rendimiento de cuatro algoritmos de aprendizaje por refuerzo (Value Iteration, Estimación Directa / Model-Based, Q-Learning y REINFORCE) en el entorno `FrozenLake-v1` bajo distintos grados de estocasticidad (`success_rate` ∈ {0.20, 0.33, 0.60, 0.80, 1.00}) y distintos tamaños de mapa (4x4, 8x8, 10x10, 12x12). Todos los resultados son medias sobre 5 seeds (3 seeds en los mapas 10x10 y 12x12 por coste computacional) y se han computado las desviaciones típicas correspondientes para acotar la varianza inter-seed.

---

## 1. Resumen general por algoritmo

### 1.1 Value Iteration

Value Iteration se comporta como un algoritmo de planificación offline puro y exhibe las propiedades teóricas esperadas:

- **Convergencia garantizada y baja varianza**. En todas las configuraciones de 4x4 la desviación típica entre seeds es ≤1.5 puntos porcentuales, lo que confirma que la política obtenida es invariante a la inicialización aleatoria del entorno: el modelo $P(s'|s,a)$ es accesible directamente, de modo que la iteración de Bellman converge al mismo punto fijo $V^*$ independientemente de la seed.
- **Inmunidad al cuello de botella exploratorio**. En 8x8 con `success_rate=0.66` mantiene un 89.3 % de éxito y en 4x4 determinista (`sr=1.0`) alcanza el 100 % con varianza nula. En estos escenarios todos los algoritmos *Model-Free* del estudio quedan al 0 % por la combinación de *sparse reward* y exploración insuficiente; Value Iteration evita el problema al no depender de explorar trayectorias para construir su política.
- **Tiempos de cómputo despreciables comparados con los métodos *Model-Free***. En 4x4 con `sr=0.33` Value Iteration converge en ~228 iteraciones internas en ≈0.025 s; en 8x8 con `sr=0.33` en ~253 iteraciones (≈0.12 s); en 12x12 sube a ~0.25 s. La complejidad escala polinómicamente con $|S|\cdot|A|$ tal y como predice la teoría, no exponencialmente.

### 1.2 Estimación Directa (Model-Based)

El método Model-Based estima $\hat{\mathcal{T}}(s'|s,a)$ y $\hat{\mathcal{R}}(s,a,s')$ a partir de transiciones muestreadas, y a continuación aplica Value Iteration sobre el modelo estimado. Su rendimiento depende críticamente de la calidad del muestreo:

- **En 4x4 con `sr=0.80` rinde mejor que Q-Learning** (83.1 % frente a 72.6 %). Esto es coherente con la teoría: cuando la dinámica subyacente es relativamente simple y la exploración consigue cubrir el espacio de estados, un método basado en modelo aprovecha mejor cada transición observada (mayor *sample efficiency*) que un método TD que actualiza una sola celda por paso.
- **Degradación pronunciada con alta estocasticidad**. Con `sr=0.33` la media baja al 58.5 % y la desviación típica sube a ~29 puntos. La estimación de $\hat{\mathcal{T}}$ requiere un número combinatoriamente mayor de visitas a cada par $(s,a)$ cuando la transición es muy ruidosa, y con un presupuesto fijo de trayectorias el modelo estimado pierde precisión.
- **Sensibilidad al número de trayectorias muestreadas**. El barrido de `num_transitions_mb` revela que con 1000 transiciones la media cae al 9 %; con 5000 se sitúa en torno al 58–59 %; con 100000 alcanza el 59.8 %. Hay un retorno decreciente claro a partir de las 5000 transiciones, lo que sugiere que el cuello de botella en 4x4 estocástico no es la cantidad de muestras sino la cobertura del espacio (estados poco frecuentes que rara vez se visitan).
- **Política aproximadamente óptima en 4x4**. El indicador `eval_policy_agreement_vi` muestra que el Model-Based coincide en un 88–96 % de los estados con la política de Value Iteration en 4x4, evidencia de que la estimación del MDP, cuando converge, recupera bien la dinámica.

### 1.3 Q-Learning

Q-Learning es el representante de la familia *Model-Free*, *Off-policy*, *Value-Based* basado en diferencias temporales. Sus resultados son los más sensibles a la parametrización:

- **Robustez moderada en 4x4 con estocasticidad variable**. La media oscila entre 68–73 % en `sr` ∈ {0.20, 0.33, 0.60} y sube a 72.6 % en `sr=0.80`. La regla de actualización TD propaga eficientemente la utilidad desde estados terminales positivos a estados intermedios cuando hay suficientes trayectorias exitosas en la fase de exploración.
- **Mejor que Model-Based bajo alta estocasticidad** (68 % vs 58 % en `sr=0.33`). Q-Learning no necesita construir explícitamente $\hat{\mathcal{T}}$, lo que evita la penalización por la cobertura incompleta del espacio. La actualización por TD aprende directamente $Q^*$ aunque la estimación implícita de la transición sea ruidosa.
- **Coincidencia con la política óptima del 85–88 % en 4x4**, ligeramente inferior al Model-Based. La diferencia se explica porque Q-Learning solo actualiza un par $(s,a)$ por paso, mientras que el Model-Based aprovecha toda la información del modelo estimado con cada barrido de Value Iteration.

### 1.4 REINFORCE

REINFORCE, como representante de los métodos de gradiente de política Monte Carlo (*on-policy*, *policy-based*), es el algoritmo más vulnerable del estudio:

- **Varianza intrínseca elevada**. La media en 4x4 `sr=0.33` con 5 seeds es 12.3 % y la desviación típica de 6.9 puntos. Esto es coherente con la teoría: el estimador del gradiente $\nabla_\theta J(\theta) \propto \sum_t G_t \nabla_\theta \log \pi(a_t|s_t;\theta)$ multiplica la verosimilitud por la ganancia completa del episodio, lo que arrastra toda la varianza de la trayectoria sin un *baseline* que la mitigue.
- **Mejora monotónica con `success_rate`**: 12.7 % → 12.3 % → 34.3 % → 62.0 % al pasar de `sr=0.20` a `sr=0.80`. Cuando el entorno es más determinista, las trayectorias exitosas se vuelven más informativas (menor varianza del retorno $G_t$ condicionada a alcanzar la meta) y el gradiente apunta más consistentemente hacia la política óptima.
- **Sensibilidad fuerte al *learning rate***. La calibración muestra que `reinforce_lr=0.001` da 17.9 %, mientras que valores más agresivos (0.1) caen al 8 %. Los gradientes de policy gradient son ruidosos y un *lr* muy alto desestabiliza la actualización antes de que la política tenga oportunidad de mejorar.
- **Coincidencia con la política óptima del 63–71 % en 4x4**, sensiblemente inferior al resto. Esto refuerza la interpretación: incluso cuando REINFORCE alcanza tasas de éxito decentes, lo hace siguiendo una política subóptima que aprovecha la estocasticidad del entorno más que un camino verdaderamente eficiente.

---

## 2. Análisis por criterio de evaluación de la práctica

### 2.1 Porcentaje de éxito en evaluación

La tabla de medias sobre 5 seeds en el experimento principal de estocasticidad es la siguiente:

| Mapa | `sr` | Value Iteration | Model-Based | Q-Learning | REINFORCE |
|------|------|-----------------|-------------|------------|-----------|
| 4x4 | 0.20 | 72.0 % | 55.0 % | 69.7 % | 12.7 % |
| 4x4 | 0.33 | 73.0 % | 58.5 % | 68.3 % | 12.3 % |
| 4x4 | 0.60 | 72.9 % | 66.8 % | 69.4 % | 34.3 % |
| 4x4 | 0.80 | 85.6 % | 83.1 % | 72.6 % | 62.0 % |
| 4x4 | 1.00 | 100.0 % | 100.0 % | 80.0 % | 60.0 % |
| 8x8 | 0.20 | 52.8 % | 0.0 % | 0.0 % | 3.9 % |
| 8x8 | 0.33 | 63.4 % | 0.0 % | 0.0 % | 0.1 % |
| 8x8 | 0.60 | 89.4 % | 9.3 % | 0.0 % | 13.7 % |
| 8x8 | 0.80 | 89.3 % | 0.0 % | 0.0 % | 14.6 % |
| 8x8 | 1.00 | 100.0 % | 0.0 % | 0.0 % | 0.0 % |

**Observaciones clave**: en 4x4 todos los algoritmos completan la tarea con tasas razonables y la jerarquía es estable (Value Iteration ≥ Model-Based ≥ Q-Learning ≫ REINFORCE). En 8x8, en cambio, solo Value Iteration mantiene su rendimiento, y los métodos *Model-Free* colapsan al 0 % por la combinación letal de mayor diámetro del MDP y *sparse reward*.

### 2.2 Evolución de la tasa de éxito durante el entrenamiento

El barrido del número de episodios (`cal_B`) en Q-Learning muestra la curva de convergencia esperada en TD:

| Episodios | Q-Learning (media) | REINFORCE (media) |
|-----------|--------------------|--------------------|
| 500 | 28.4 % | 6.1 % |
| 1000 | 64.5 % | 17.7 % |
| 2000 | 68.3 % | 12.3 % |
| 5000 | 74.3 % | 17.0 % |
| 10000 | 71.6 % | 31.0 % |
| 20000 | 73.1 % | 46.3 % |

Q-Learning satura su tabla $Q$ alrededor de los 5000 episodios y a partir de ahí las mejoras son marginales: la propagación de Bellman ha llegado al límite que permite el `epsilon` actual. REINFORCE, en cambio, sigue mejorando lentamente hasta los 20000 episodios y aún no ha saturado, reflejando su *sample inefficiency* característica.

### 2.3 Tiempo de entrenamiento y tiempo de convergencia

| Mapa | `sr` | VI | MB | QL | RF |
|------|------|------|------|------|------|
| 4x4 | 0.33 | 0.025 s (228 iter) | 0.050 s | 0.44 s | 0.62 s |
| 4x4 | 1.00 | 0.001 s (7 iter) | 0.018 s | 0.33 s | 0.67 s |
| 8x8 | 0.33 | 0.118 s (253 iter) | 0.012 s | 1.56 s | 2.62 s |
| 8x8 | 0.66 | 0.094 s (206 iter) | 0.025 s | 2.01 s | 2.81 s |
| 12x12 | 0.33 | 0.246 s (235 iter) | 0.013 s | 1.40 s | 1.96 s |

Value Iteration escala polinomialmente con $|S|$: de 0.025 s en 4x4 a 0.25 s en 12x12 (factor ~10 al multiplicar por ~9 el número de estados, coherente con $O(|S|^2|A|)$ por iteración). Los métodos *Model-Free* tienen tiempos dominados por el número de episodios y los pasos por episodio, no por la estructura del MDP, por lo que aunque sus tiempos son ~50–100 veces mayores que VI, no explotan exponencialmente con el tamaño. El número de iteraciones internas hasta convergencia de VI aumenta con la estocasticidad (de 7 con `sr=1.0` a 228 con `sr=0.33` en 4x4) porque mayor ruido implica que la magnitud de las actualizaciones decae más lentamente entre iteraciones.

### 2.4 Optimalidad de la política en entornos estocásticos

El indicador `eval_policy_agreement_vi` (porcentaje de estados en los que la política aprendida coincide con la de Value Iteration) ofrece una medida directa de optimalidad:

| Mapa | `sr` | Model-Based | Q-Learning | REINFORCE |
|------|------|-------------|------------|-----------|
| 4x4 | 0.33 | 88.8 % | 85.0 % | 63.8 % |
| 4x4 | 0.60 | 91.2 % | 88.8 % | 67.5 % |
| 4x4 | 0.80 | 96.2 % | 85.0 % | 71.2 % |
| 8x8 | 0.60 | 36.6 % | 29.7 % | 45.9 % |
| 8x8 | 0.80 | 26.6 % | 26.9 % | 45.6 % |

En 4x4 el Model-Based recupera la política óptima en más del 88 % de los estados, Q-Learning en torno al 85 % y REINFORCE en torno al 65–71 %. En 8x8 todos los métodos *Model-Free* pierden el control: aunque REINFORCE alcanza algunos éxitos esporádicos, su política está más cerca del azar que de $\pi^*$. Curiosamente REINFORCE supera a Q-Learning en agreement bajo 8x8 a pesar de tener tasas de éxito similares: la política estocástica de REINFORCE asigna alguna probabilidad a casi todas las acciones, lo que en agregado se parece más a una política uniforme que la política codiciosa pero mal entrenada de Q-Learning.

---

## 3. Análisis por hiperparámetro

### 3.1 Factor de descuento $\gamma$

| $\gamma$ | VI | Q-Learning | REINFORCE |
|----------|------|------------|-----------|
| 0.90 | 73.1 % | 62.5 % | 29.8 % |
| 0.95 | 73.1 % | 67.7 % | 27.9 % |
| 0.99 | 73.0 % | 71.6 % | 46.3 % |

**Value Iteration**: el resultado es prácticamente independiente de $\gamma$ en 4x4. La ordenación relativa de acciones bajo la ecuación de Bellman es la misma para $\gamma \in [0.90, 0.99]$ en este mapa, de modo que la política resultante converge al mismo argmax. Lo que sí varía es el número de iteraciones hasta convergencia: 60 con $\gamma=0.9$, 98 con $\gamma=0.95$ y 228 con $\gamma=0.99$. Esto es coherente con la teoría: el factor de contracción de la operación de Bellman es $\gamma$, por lo que valores próximos a 1 ralentizan la convergencia.

**Q-Learning**: muestra una preferencia clara por $\gamma$ alto (de 62.5 % a 71.6 % al subir de 0.9 a 0.99). En 4x4 con `sr=0.33` el agente necesita reconocer recompensas diferidas muchos pasos por delante (camino de ~6 pasos con probabilidad alta de resbalar); un $\gamma$ bajo penaliza desproporcionadamente las trayectorias largas y empuja al agente hacia políticas miopes.

**REINFORCE**: también mejora con $\gamma$ alto (de 29.8 % a 46.3 %), pero con altísima varianza inter-seed.

### 3.2 Coeficiente de exploración $\epsilon$ (Q-Learning)

| $\epsilon$ | Q-Learning (media) | Desviación típica |
|------------|--------------------|--------------------|
| 0.3 | 14.8 % | 29.6 |
| 0.5 | 57.7 % | 28.9 |
| 0.8 | 73.3 % | 1.9 |
| 1.0 | 68.3 % | 5.7 |

El comportamiento bimodal con $\epsilon$ bajo es revelador: con $\epsilon=0.3$ cuatro seeds quedan al 0 % y una alcanza el 74 %. Esto refleja directamente la condición de optimalidad del algoritmo: toda combinación $(s,a)$ debe visitarse un número infinito de veces. Con $\epsilon$ bajo y `epsilon_decay=0.999`, el agente se vuelve codicioso sobre una tabla $Q$ casi vacía antes de descubrir la meta, y queda atrapado en una política inicial subóptima. $\epsilon=0.8$ proporciona el equilibrio óptimo entre exploración suficiente y explotación de la información aprendida.

### 3.3 Tasa de aprendizaje $\alpha$ (Q-Learning)

| $\alpha$ | Q-Learning (media) | Desviación típica |
|----------|--------------------|--------------------|
| 0.05 | 56.6 % | 18.8 |
| 0.10 | 73.3 % | 1.9 |
| 0.30 | 55.5 % | 19.3 |
| 0.50 | 52.2 % | 16.1 |

$\alpha=0.10$ es el punto dulce. Con $\alpha=0.05$ la propagación de Bellman es demasiado lenta para alcanzar la convergencia en el número de episodios disponibles; con $\alpha \geq 0.30$ el algoritmo asigna demasiado peso a la muestra más reciente y oscila ante la estocasticidad del entorno, perdiendo la estabilidad necesaria para acumular información a largo plazo. Esto valida empíricamente la condición de Robbins–Monro $\sum_k \alpha_k = \infty, \sum_k \alpha_k^2 < \infty$ vista en teoría: tasas demasiado altas violan la segunda condición y rompen la convergencia.

### 3.4 *Learning rate* de REINFORCE

| `reinforce_lr` | REINFORCE (media) |
|----------------|--------------------|
| 0.001 | 17.9 % |
| 0.01 | 12.3 % |
| 0.05 | 20.4 % |
| 0.10 | 8.1 % |

Para REINFORCE no hay una tendencia monotónica clara, sino que todos los valores producen resultados pobres con alta varianza. El gradiente de policy gradient es intrínsecamente ruidoso, y ningún `lr` puede compensarlo sin un *baseline* (lo que llevaría a una arquitectura Actor-Critic, fuera del alcance de esta práctica).

### 3.5 Señal de recompensa (*reward shaping*)

| Configuración | Q-Learning | REINFORCE |
|---------------|------------|-----------|
| Por defecto (1.0 meta, 0.0 resto) | 68.3 % | 12.3 % |
| Hole penalty (-1.0 en agujeros) | 69.6 % | 6.8 % |
| Step penalty (-0.01 por paso) | 52.7 % | 4.2 % |

El **hole penalty** mejora marginalmente Q-Learning (de 68 a 70 %) y empeora REINFORCE (de 12 a 7 %). Para Q-Learning la penalización proporciona una señal informativa al actualizar transiciones a estados terminales negativos, acelerando la propagación de utilidades. Para REINFORCE las trayectorias completas tienen ahora ganancias $G_t$ negativas grandes en los fallos, lo que aumenta aún más la varianza del gradiente.

El **step penalty** degrada ambos algoritmos: en Q-Learning baja a 52.7 % con varianza muy alta entre seeds; en REINFORCE colapsa a 4.2 %. En un entorno con horizontes largos por la estocasticidad, la penalización por paso convierte cualquier camino largo en una pérdida acumulada, y aparece un fenómeno clásico de *reward hacking*: el agente prefiere terminar el episodio cayendo en un agujero a seguir sumando penalizaciones por paso. La recompensa por defecto, aunque *sparse*, es preferible.

---

## 4. Escalabilidad

### 4.1 Tamaño del mapa con `sr=0.33`

| Mapa | $|S|$ | Value Iteration | Model-Based | Q-Learning | REINFORCE |
|------|------|-----------------|-------------|------------|-----------|
| 4x4 | 16 | 73.0 % | 58.5 % | 68.3 % | 12.3 % |
| 8x8 | 64 | 63.4 % | 0.0 % | 0.0 % | 0.1 % |
| 10x10 | 100 | 14.9 % | 0.0 % | 0.0 % | 0.0 % |
| 12x12 | 144 | 20.3 % | 0.0 % | 0.0 % | 0.0 % |

Value Iteration es el único algoritmo que mantiene tasas de éxito por encima del azar en mapas grandes. La caída del 63 % al 15 % entre 8x8 y 10x10 no es por fallo del algoritmo sino por la propia dificultad del MDP: en mapas grandes con `sr=0.33` el camino esperado a la meta es muy largo y la probabilidad acumulada de caer en un agujero por resbalones tiende a 1. Incluso la política óptima ofrece tasas de éxito limitadas por el propio entorno.

Los métodos *Model-Free* sufren un colapso total: la combinación de *sparse reward* (única señal positiva al alcanzar una meta a 14+ pasos) con la exploración $\epsilon$-greedy hace que la probabilidad de tropezar por casualidad con la meta durante el entrenamiento sea ínfima. Mientras eso no ocurre, todas las actualizaciones TD y todos los gradientes de política son nulos o ruido, manteniendo al agente en una caminata aleatoria ciega.

### 4.2 Tiempo de cómputo vs porcentaje de éxito

Value Iteration es Pareto-dominante en todos los escenarios: gana en tiempo de cómputo (10–100× más rápido que los *Model-Free*) y simultáneamente en tasa de éxito, incluso en mapas pequeños donde los *Model-Free* deberían ser competitivos. Esto ilustra el siguiente principio práctico: **cuando el MDP es accesible, planificar es estrictamente preferible a aprender por interacción**.

---

## 5. Recomendaciones prácticas

1. **Si el MDP es accesible, usar Value Iteration**. No existe ningún escenario en el estudio donde un método *Model-Free* mejore a Value Iteration cuando éste tiene acceso al modelo. La inversión en construir o consultar el MDP se amortiza muy rápidamente.

2. **Si el MDP no es accesible pero el espacio de estados es pequeño**, el orden de preferencia es Model-Based > Q-Learning > REINFORCE. Model-Based aprovecha mejor cada transición y consigue políticas más cercanas a la óptima.

3. **Si el espacio de estados crece**, ningún método tabular puro funcionará bajo *sparse rewards*. Las direcciones a explorar son: introducir aproximadores de función (DQN, redes neuronales), introducir un *replay buffer*, o emplear arquitecturas Actor-Critic que reduzcan la varianza de REINFORCE mediante un crítico que estime $V^\pi(s)$.

4. **Calibración de $\epsilon$**: mantener $\epsilon \geq 0.8$ con decaimiento gradual al menos hasta que la $Q$-table tenga cobertura suficiente. Valores bajos producen comportamientos bimodales (o converge o no, dependiendo de la seed).

5. **Calibración de $\alpha$**: $\alpha=0.1$ es el punto dulce en este tipo de tareas tabulares con horizontes moderados. Valores más altos rompen las condiciones de Robbins–Monro y desestabilizan la convergencia.

6. **No modificar la señal de recompensa salvo justificación fuerte**. El *step penalty* hunde el rendimiento; el *hole penalty* solo ayuda marginalmente a Q-Learning. La recompensa *sparse* por defecto es preferible en la mayoría de los casos.

7. **Sobre la varianza inter-seed**: cualquier conclusión basada en una única seed es estadísticamente cuestionable. Los métodos *Model-Free* exhiben varianzas inter-seed de hasta ±30 puntos en algunas configuraciones; sin múltiples seeds es imposible distinguir un resultado real de un artefacto.

---

## 6. Propuestas de visualizaciones (sugerencias, no obligatorias)

*Las siguientes ideas son sugerencias para ilustrar los resultados de manera explicativa en la entrega final. No son obligatorias y se pueden combinar, descartar o sustituir según las necesidades del informe.*

### 6.1 Gráfico principal: comparativa por algoritmo y configuración

- **Barras agrupadas con barras de error** mostrando la media ± desviación típica sobre las 5 seeds. Eje X: success_rate; un grupo de cuatro barras por nivel (un color por algoritmo). Una figura por mapa (4x4 y 8x8). Permite visualizar simultáneamente el rendimiento absoluto, la varianza inter-seed y el efecto de la estocasticidad.

### 6.2 Escalabilidad

- **Gráfico de líneas en escala doble eje**: eje X con tamaño de mapa (4x4, 8x8, 10x10, 12x12), eje Y izquierdo con tasa de éxito (%) y eje Y derecho con tiempo de entrenamiento (s) en escala logarítmica. Una línea por algoritmo. Hace visible el trade-off coste vs rendimiento y la caída abrupta de los métodos *Model-Free*.

### 6.3 Convergencia episódica

- **Curvas de aprendizaje suavizadas** (media móvil de ~100 episodios) mostrando la evolución de la tasa de éxito acumulada durante el entrenamiento de Q-Learning y REINFORCE. Sombreado del intervalo $\mu \pm \sigma$ sobre las 5 seeds. Permite contrastar visualmente la convergencia estable de Q-Learning con las oscilaciones de REINFORCE.

### 6.4 Calibración de hiperparámetros

- **Heatmap o gráfico de líneas con dos paneles**: panel izquierdo con barrido de $\epsilon$ y panel derecho con barrido de $\alpha$ para Q-Learning. Cada celda o punto incluye la media y la varianza. La bimodalidad con $\epsilon=0.3$ es particularmente didáctica si se muestran los 5 valores individuales superpuestos en lugar de solo la media.

### 6.5 Heatmap de optimalidad

- **Heatmap del mapa 4x4 o 8x8** coloreado por el porcentaje de seeds que coinciden con la acción óptima de Value Iteration en cada celda. Permite ver geográficamente dónde fallan los métodos *Model-Free* (típicamente en los estados cercanos a agujeros y lejanos a la meta).

### 6.6 Visualización de la política aprendida

- **Mapa 4x4 con flechas** indicando la acción que toma cada algoritmo en cada estado. Comparar las políticas de los cuatro algoritmos lado a lado para `sr=0.33` muestra visualmente cómo la política óptima de Value Iteration evita las celdas próximas a agujeros mientras que REINFORCE produce políticas menos coherentes.

### 6.7 Distribución de seeds (resumen estadístico)

- **Boxplots o violin plots** por algoritmo y configuración, mostrando la distribución completa de las 5 seeds en lugar de solo la media. Especialmente útil para comunicar la varianza intrínseca de REINFORCE y la bimodalidad de Q-Learning con $\epsilon$ bajo.

### 6.8 Reward shaping

- **Gráfico de barras agrupadas comparando las tres configuraciones de recompensa** (default, hole penalty, step penalty) en Q-Learning y REINFORCE. Permite cuantificar el impacto y justificar la recomendación de mantener la recompensa por defecto.
