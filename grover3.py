from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

SECRET = "101"  # 8 boxes now: "000" ... "111"
ROUNDS = 1     # how many times to repeat oracle + diffuser

QUBITS = [0, 1, 2]

def flip_zeros(qc, secret):
    """X on every qubit whose secret bit is 0, so the secret becomes '111'."""
    for i, bit in enumerate(reversed(secret)):  # qubit 0 is on the right
        if bit == "0":
            qc.x(i)

def oracle(qc, secret):
    """Minus sign on the secret answer."""
    flip_zeros(qc, secret)
    qc.ccz(0, 1, 2)       # minus sign on '111' only
    flip_zeros(qc, secret)

def diffuser(qc):
    """Flip every weight around the average."""
    qc.h(QUBITS)
    qc.x(QUBITS)
    qc.ccz(0, 1, 2)
    qc.x(QUBITS)
    qc.h(QUBITS)

qc = QuantumCircuit(3)
qc.h(QUBITS)            # blend all 8 answers (12.5% each)
for _ in range(ROUNDS):
    oracle(qc, SECRET)
    diffuser(qc)

probs = Statevector(qc).probabilities_dict()
print(f"Rounds: {ROUNDS}, chance of finding {SECRET}: {probs[SECRET]:.1%}")