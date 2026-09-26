from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt



qc = QuantumCircuit(1)   # one qubit, starts at 0

# 1. qc.h(0) then qc.s(0). Where does the j show up? Do the probabilities change?
qc.h(0), qc.s(0)

#2. qc.h(0), qc.s(0), qc.s(0). Two quarter-turns make a half-turn. Which gate from lesson 2 does this match?
# qc.h(0), qc.s(0), qc.s(0)

# 3. qc.h(0), qc.s(0), qc.s(0), qc.h(0). Predict the result before you run it!
# qc.h(0), qc.s(0), qc.s(0), qc.h(0)

#counts = Statevector(qc).sample_counts(1000)  # measure 1000 times
#print(counts)
print(Statevector(qc))
print(Statevector(qc).probabilities_dict())
qc.draw("mpl")
plt.show()