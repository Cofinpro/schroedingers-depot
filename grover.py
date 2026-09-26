from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# Gate	Code	Rule on the weights	In words
# X	qc.x(0)	(a, b) → (b, a)	Quantum NOT: swap 0 and 1
# H	qc.h(0)	(a, b) → (0.707·(a+b), 0.707·(a−b))	Makes blends, and un-makes them (Hadamard  gate)
# Z	qc.z(0)	(a, b) → (a, −b)	Half-turn on the 1-weight (invisible sign)
# S	qc.s(0)	(a, b) → (a, j·b)	Quarter-turn on the 1-weight
# CX	qc.cx(0, 1)	if qubit 0 is 1, flip qubit 1	Swaps the weights of '01' and '11'
# CZ	qc.cz(0, 1)	minus sign on '11' only	Z for two qubits (Grover's oracle)

# Handy combos you discovered:
#
# H H = do nothing
# S S = Z
# H Z H = X (a hidden sign becomes a visible flip)

SECRET = "10"  # hide the prize: try "00", "01", "10" or "11"

def oracle(qc, secret):
    """Put a minus sign on the secret answer."""
    # cz puts a minus on '11' only, so X gates turn the secret into '11' first
    for i, bit in enumerate(reversed(secret)):  # reversed: qubit 0 is on the right
        if bit == "0":
            qc.x(i)
    qc.cz(0, 1)
    for i, bit in enumerate(reversed(secret)):  # undo the X gates
        if bit == "0":
            qc.x(i)

def diffuser(qc):
    """Interference: turn the hidden minus sign into a big weight."""
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])

qc = QuantumCircuit(2)
qc.h([0, 1])        # step 1: blend all 4 answers
oracle(qc, SECRET)  # step 2: mark the secret
diffuser(qc)        # step 3: amplify it

print(qc.draw())
probs = Statevector(qc).probabilities_dict()
print({str(k): round(float(v), 3) for k, v in probs.items()})
Statevector(qc).draw('bloch')
qc.draw("mpl")
plt.show()