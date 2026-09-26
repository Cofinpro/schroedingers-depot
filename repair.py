from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

ERROR_ON = 1   # which qubit gets hit by noise: 0, 1, 2, or None for no error

qc = QuantumCircuit(3)

# 1. The precious qubit (qubit 0): a 20/80 blend
qc.ry(2.214, 0)             # tip the arrow part of the way down

# 2. Encode: entangle it with 2 helper qubits
qc.cx(0, 1)
qc.cx(0, 2)

# 3. Noise strikes: an accidental X on one qubit
if ERROR_ON is not None:
    qc.x(ERROR_ON)

# 4. Decode and repair: majority vote
qc.cx(0, 1)
qc.cx(0, 2)
qc.ccx(2, 1, 0)             # if helpers 1 AND 2 both disagree, qubit 0 was the bad one: flip it back

odds = Statevector(qc).probabilities([0])   # look at qubit 0 only
print(f"error on {ERROR_ON}: qubit 0 is 0 with {odds[0]:.0%}, 1 with {odds[1]:.0%}")