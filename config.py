# config.py aqui las variables para poder cambiar los parametros
import numpy as np

# --- 1. CONFIGURACIÓN DEL ENTORNO ---
SLIPPERY = True # el mapa resbala
MAP_NAME = "4x4" # tamaño del mapa (puede ser 8x8)
SUCCESS_RATE = 0.8 # probabilidad de que la accion se lleve a cabo
SEED = 99 # semilla fija del mapa para las pruebas

# --- 2. REWARD SHAPING ---
DEFAULT_REWARD = False
HOLE_PENALTY = -0.5
STEP_PENALTY = 0.0
GOAL_REWARD = 1.0

# --- 3. HIPERPARÁMETROS GENERALES ---
GAMMA = 0.99 # factor descuento
T_MAX = 100 # maximo de pasos por episodio
NUM_EPISODES = 2000 # numero de pruebas para entrenar al agente

LEARNING_RATE = 0.1 # tasa de aprendizaje (parametro alfa)
LR_DECAY = 0.999 # descuento de la tasa de aprendizaje
EPSILON = 0.2 # exploracion inicial
EPSILON_DECAY = 0.995 # descuento epsilon
EPSILON_MIN = 0.01 # valor minimo exploracion

THETA_CONVERGENCE = 1e-6 # umbral de convergencia diferencia de valor

REINFORCE_LR = 0.1 # learning rate del reinforce

NUM_EPISODES_TEST = 100 # tests finales que se hacen para comparar los algoritmos
REWARD_THRESHOLD = 0.9 # umbral para considerar el problema resuelto