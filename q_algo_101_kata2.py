from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt



qc = QuantumCircuit(1)   # one qubit, starts at 0
#qc.x(0)                  # put it in superposition
#qc.h(0)
#qc.z(0)
#qc.h(0)

qc.h(0)   # first: make the 50/50 blend
qc.z(0)   # then: put a minus on the weight for 1

counts = Statevector(qc).sample_counts(1000)  # measure 1000 times
#print(counts)
print(Statevector(qc))
print(Statevector(qc).probabilities_dict())
qc.draw("mpl")
plt.show()