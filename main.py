import config
from environment import create_env
import numpy as np

def evaluate_agent(agent, env, num_episodes=None):
    """
    Evalúa el rendimiento de un agente sin aprender (modo ejecución).
    Utiliza NUM_EPISODES_TEST definido en config.py.
    """
    episodes = num_episodes if num_episodes else config.NUM_EPISODES_TEST
    successes = 0
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0

        # El bucle se detiene por estado terminal, truncamiento o límite de pasos
        while not (done or truncated) and steps < config.T_MAX:
            # Seleccionamos la mejor acción sin exploración (training=False)
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            
            # Verificamos si ha alcanzado la meta (recompensa positiva) 
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
    
    # 1. Creamos el entorno
    env = create_env(render_mode=None)

    # 2. Creamos el agente de Iteración de Valor
    agent = ValueIterationAgent(env)

    # 3. ENTRENAMIENTO (En Value Iteration es el cálculo de la tabla V)
    print("Entrenando agente (Cálculo de utilidades)...")
    start_time = time.time()
    iterations = agent.train() # El método que programaste antes
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Convergencia alcanzada en {iterations} iteraciones.")
    print(f"Tiempo de cómputo: {training_time:.4f} segundos.")

    # 4. EVALUACIÓN
    print(f"Evaluando política en {config.NUM_EPISODES_TEST} episodios...")
    success_rate, avg_reward = evaluate_agent(agent, env)

    # 5. MOSTRAR RESULTADOS
    print("\n" + "="*30)
    print(f"RESULTADOS FINALES - {config.MAP_NAME}")
    print(f"Porcentaje de éxito: {success_rate:.2f}%")
    print(f"Recompensa promedio: {avg_reward:.4f}")
    print("="*30)

    env.close()