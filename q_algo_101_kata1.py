from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(1)   # one qubit, starts at 0
qc.h(0)                  # put it in superposition
#qc.h(0)

#counts = Statevector(qc).sample_counts(1000)  # measure 1000 times
#print(counts)
print(Statevector(qc))
print(qc.draw())
probs = Statevector(qc).probabilities_dict()
print({str(k): round(float(v), 3) for k, v in probs.items()})