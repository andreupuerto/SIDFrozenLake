import gymnasium as gym
from gymnasium import Wrapper

class CustomFrozenLakeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Ejemplo de modificación de recompensa (Notebook 5):
        # Penalizar caer en un agujero o tardar mucho tiempo [cite: 34]
        if terminated and reward < 1:
            reward = -1.0  # Penalización por caer al agua
        elif not terminated:
            reward = -0.01 # Pequeña penalización por cada paso (incentiva rapidez)
            
        return state, reward, terminated, truncated, info

def create_env(map_name="4x4", is_slippery=True, render_mode=None):
    # El enunciado menciona el uso de success_rate si el entorno lo soporta [cite: 9, 19]
    env = gym.make(
        "FrozenLake-v1", 
        map_name=map_name, 
        is_slippery=is_slippery, 
        render_mode=render_mode
    )
    return CustomFrozenLakeWrapper(env)