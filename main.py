import config
from environment import create_env
import numpy as np
from reinforce import ReinforceAgent

def evaluate_agent(agent, env, num_episodes=None):
    episodes = num_episodes if num_episodes else config.NUM_EPISODES_TEST
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0

        while not (done or truncated) and steps < config.T_MAX:
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            
            if done and reward >= 1.0:
                successes += 1
        
        total_rewards.append(ep_reward)

    success_rate = (successes / episodes) * 100
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_reward

if __name__ == "__main__":
    import time
    from value_iteration import ValueIterationAgent # Importamos tu clase

    print(f"--- Iniciando Experimento: {config.MAP_NAME} (Slippery={config.SLIPPERY}) ---")
    
    env = create_env(render_mode=None)

    #AGENTES (se tendra que añadir un metodo para elegir)
    # agent = ValueIterationAgent(env)
    agent = ReinforceAgent(env)
    
    print("Entrenando agente (Cálculo de utilidades)...")
    start_time = time.time()
    iterations = agent.train()
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Convergencia alcanzada en {iterations} iteraciones.")
    print(f"Tiempo de cómputo: {training_time:.4f} segundos.")

    print(f"Evaluando política en {config.NUM_EPISODES_TEST} episodios...")
    success_rate, avg_reward = evaluate_agent(agent, env)

    print("\n" + "="*30)
    print(f"RESULTADOS FINALES - {config.MAP_NAME}")
    print(f"Porcentaje de éxito: {success_rate:.2f}%")
    print(f"Recompensa promedio: {avg_reward:.4f}")
    print("="*30)

    env.close()