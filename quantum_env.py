from qiskit import QuantumCircuit 
from qiskit import Aer, execute
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

class QuantumCircuitEnv:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.reset()

    def reset(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        self.state = self.get_state()
        return self.state

    def get_state(self):
        state_representation = len(self.circuit.data)
        return state_representation

    def step(self, action):
        if action == 0:
            self.circuit.h(0)
        elif action == 1:
            self.circuit.x(0)
        
        next_state = self.get_state()
        reward = self.compute_reward()
        done = self.is_done()
        return next_state, reward, done, {}

    def compute_reward(self):
        reward = -1  # Negative reward for each step
        if len(self.circuit.data) > 10:
            reward -= 10  # Larger negative reward if the circuit exceeds 10 gates
        return reward

    def is_done(self):
        return len(self.circuit.data) >= 20  # Terminate after 20 gates

# Create the environment
env = QuantumCircuitEnv(num_qubits=1)
