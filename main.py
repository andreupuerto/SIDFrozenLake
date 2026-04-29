import config
import time
import numpy as np
import matplotlib.pyplot as plt
from environment import create_env
from value_iteration import ValueIterationAgent
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

def plot_learning(history, name):
    plt.figure(figsize=(10, 5))
    plt.plot(history, alpha=0.3, color='blue', label="Recompensa")
    if len(history) > 100:
        smooth = np.convolve(history, np.ones(100)/100, mode='valid')
        plt.plot(smooth, color='red', label="Media móvil (100 ep.)")
    plt.title(f"Progreso de Entrenamiento - {name} ({config.MAP_NAME})")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Total")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*35)
    print("   SISTEMA DE CONTROL DE AGENTES")
    print("="*35)
    print(f"Config: Mapa {config.MAP_NAME} | Slippery={config.SLIPPERY}")
    print("-" * 35)
    print("1. Value Iteration (Model-Based)")
    print("2. REINFORCE (Policy Gradient)")
    
    opcion = input("\nSelecciona el algoritmo a ejecutar: ")

    env = create_env()
    
    if opcion == '1':
        agent = ValueIterationAgent(env)
    else:
        agent = ReinforceAgent(env)

    print(f"\nEntrenando {agent.__class__.__name__}...")
    start_time = time.time()
    resultado = agent.train()
    end_time = time.time()

    print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

    if isinstance(resultado, list):
        plot_learning(resultado, agent.__class__.__name__)
    else:
        print(f"Convergencia alcanzada en {resultado} iteraciones.")

    print(f"\nEvaluando en {config.NUM_EPISODES_TEST} episodios...")

    success_rate, avg_reward = evaluate_agent(agent, env)

    print("\n" + "="*30)
    print(f"RESULTADOS FINALES - {config.MAP_NAME}")
    print(f"Porcentaje de éxito: {success_rate:.2f}%")
    print(f"Recompensa promedio: {avg_reward:.4f}")
    print("="*30)

    env.close()