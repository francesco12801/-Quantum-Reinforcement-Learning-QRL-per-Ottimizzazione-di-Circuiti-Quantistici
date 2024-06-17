from agent_setup import agent 
from quantum_environment import env_set_up 
import deep_network
import numpy as np

def main():
    episodes = 100
    batch_size = 32

    # Initialize the environment
    env = env_set_up.QuantumCircuitEnv(num_qubits=1)

    # Initialize the Tabular Q-learning agent
    state_space = 11  # Simplified state space: 0-10 gates
    action_space = 2  # Two actions: add H gate or add X gate
    q_agent = agent.QLearningAgent(state_space, action_space)

    # Train the Tabular Q-learning agent
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = q_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            q_agent.learn(state, action, reward, next_state)
            state = next_state

    # Initialize the DQN agent
    dqn_agent = deep_network.DQNAgent(state_space, action_space)

    # Train the DQN agent
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 1])  # Reshape for neural network input
        done = False
        while not done:
            action = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 1])
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)

    # Evaluate the agents
    # Here you would typically run the agents without exploration and measure performance
    print("Training complete")

if __name__ == "__main__":
    main()
