from qiskit import QuantumCircuit, Aer, execute


def circuit_simulation(circuit):
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    return counts 

def main():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    counts = circuit_simulation(circuit)
    print(counts)
    