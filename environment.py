# environment.py
import gymnasium as gym
from gymnasium import Wrapper
import config

class CustomFrozenLakeWrapper(Wrapper):
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if not config.DEFAULT_REWARD:
            if terminated and reward < 1:
                reward = config.HOLE_PENALTY
            elif terminated and reward >= 1:
                reward = config.GOAL_REWARD
            elif not terminated:
                reward = config.STEP_PENALTY
                
        return state, reward, terminated, truncated, info

def create_env(map_name=None, is_slippery=None, render_mode=None):
    m_name = map_name if map_name else config.MAP_NAME
    slippery = is_slippery if is_slippery is not None else config.SLIPPERY
    
    env = gym.make("FrozenLake-v1", map_name=m_name, is_slippery=slippery,
               render_mode=render_mode, success_rate=config.SUCCESS_RATE)
    return CustomFrozenLakeWrapper(env)