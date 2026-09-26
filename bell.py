from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

def show(qc):
    print(qc.draw())
    probs = Statevector(qc).probabilities_dict()
    print("odds:   ", {str(k): round(float(v), 3) for k, v in probs.items()})
    counts = Statevector(qc).sample_counts(10)  # measure 10 times
    print("10 runs:", {str(k): int(v) for k, v in counts.items()})

# Task 1: X on qubit 0 only
qc = QuantumCircuit(2)
qc.x(0)
show(qc)
# 1

# Task 2: X on qubit 0, then CX
qc = QuantumCircuit(2)
qc.x(0)
qc.cx(0, 1)
show(qc)
# 1

# Task 3: H on qubit 0, then CX (the famous Bell state)
qc = QuantumCircuit(2)
qc.x(1)
qc.h(0)
qc.cx(0, 1)
show(qc)
Statevector(qc).draw('bloch')
qc.draw("mpl")
plt.show()