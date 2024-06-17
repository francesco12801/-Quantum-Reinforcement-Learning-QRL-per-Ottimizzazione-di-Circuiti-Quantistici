
import DQN,Q_learn_agent,quantum_env 
import numpy as np
import matplotlib.pyplot as plt

def main():
    episodes = 100  # Increase number of episodes
    batch_size = 32

    # Initialize the environment
    env = quantum_env.QuantumCircuitEnv(num_qubits=1)

    # Initialize the Tabular Q-learning agent
    state_space = 21  # Simplified state space: 0-20 gates
    action_space = 2  # Two actions: add H gate or add X gate
    q_agent = Q_learn_agent.QLearningAgent(state_space, action_space)

    # Initialize lists to store rewards
    q_learning_rewards = []

    # Train the Tabular Q-learning agent
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = q_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            q_agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        q_learning_rewards.append(total_reward)

    # Initialize the DQN agent
    dqn_agent = DQN.DQNAgent(state_space=1, action_space=action_space)

    # Initialize lists to store rewards
    dqn_rewards = []

    # Train the DQN agent
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 1])  # Reshape for neural network input
        done = False
        total_reward = 0
        while not done:
            action = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 1])
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)
        dqn_rewards.append(total_reward)

    # Plot the rewards
    plt.figure(figsize=(12, 6))
    plt.plot(q_learning_rewards, label='Q-learning')
    plt.plot(dqn_rewards, label='DQN')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode for Q-learning and DQN')
    plt.legend()
    plt.show()

    print("Training complete")

if __name__ == "__main__":
    main()
