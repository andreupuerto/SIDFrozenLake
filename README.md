# FrozenLake RL — Práctica 2

## Antes de empezar

Se requieren instalar las librerias necesarias para el proyecto, para ello ejecutar en terminal:
```bash
pip install pygame numpy gymnasium matplotlib
```

## Explicación

Esta práctica se centra en la implementación y comparación de diferentes algoritmos de aprendizaje por refuerzo en el entorno FrozenLake. Se ha hecho de forma modular, para facilitar la implementación de nuevos algoritmos, aunque sólo están implementados los pedidos en la teoría.

## Estructura del proyecto



## Ejecutar Experimentos

### Pruebas Libres

python main.py

### Experimentos de la práctica

# 1. Calibración de Gamma
python experiments.py --exp calibration_gamma --save

# 2. Calibración de Episodios
python experiments.py --exp calibration_episodes --save

# 3. Calibración de Recompensas
python experiments.py --exp calibration_reward --save

# 4. Calibración de Epsilon y Alpha
python experiments.py --exp calibration_epsilon --save

# 5. Experimento Principal
python experiments.py --exp main --save

# 6. Todos los experimentos en orden
python experiments.py --exp all --save
