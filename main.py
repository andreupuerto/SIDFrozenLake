# main.py  — ejecución interactiva de un único agente
import time
import config
from environment import create_env
from utils import evaluate_agent, plot_learning
from value_iteration import ValueIterationAgent
from reinforce import ReinforceAgent

def build_agent(name, env):
    if name == "value_iteration":
        return ValueIterationAgent(env)
    elif name == "reinforce":
        return ReinforceAgent(env)
    # elif name == "qlearning":
    #     return QLearningAgent(env)
    # elif name == "model_based":
    #     return ModelBasedAgent(env)
    else:
        raise ValueError(f"Agente desconocido: {name}")


def interactive_config():
    print("\n--- CONFIGURACIÓN ---")

    mapa = input(f"Mapa (4x4/8x8) [actual: {config.MAP_NAME}]: ").strip()
    if mapa in ["4x4", "8x8"]:
        config.MAP_NAME = mapa

    sr = input(f"Success rate 0-1 [actual: {config.SUCCESS_RATE}]: ").strip()
    try:
        config.SUCCESS_RATE = float(sr)
    except ValueError:
        pass

    ep = input(f"Número de episodios [actual: {config.NUM_EPISODES}]: ").strip()
    if ep.isdigit():
        config.NUM_EPISODES = int(ep)


if __name__ == "__main__":
    interactive_config()

    print("\n" + "=" * 40)
    print(f" Mapa: {config.MAP_NAME} | Success rate: {config.SUCCESS_RATE}")
    print("=" * 40)
    print("1. Value Iteration")
    print("2. REINFORCE")
    print("3. Q-Learning")
    print("4. Model Based")

    opciones = {"1": "value_iteration", "2": "reinforce", "3": "qlearning", "4": "model_based"}
    opcion = input("\nSelecciona algoritmo: ").strip()

    if opcion not in opciones:
        print("Opción no válida.")
        exit(1)

    env = create_env()
    agent = build_agent(opciones[opcion], env)

    print(f"\nEntrenando {agent.__class__.__name__}...")
    start = time.time()
    resultado = agent.train()
    elapsed = time.time() - start
    print(f"Entrenamiento completado en {elapsed:.2f}s")

    if isinstance(resultado, list):
        plot_learning(resultado, agent.__class__.__name__, config.MAP_NAME)
    else:
        print(f"Convergencia en {resultado} iteraciones.")

    success_rate, avg_reward = evaluate_agent(agent, env)
    print(f"\nÉxito: {success_rate:.2f}% | Recompensa media: {avg_reward:.4f}")
    env.close()
