from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

FIX_X = True   # Bob's flip correction
FIX_Z = True   # Bob's sign correction

qc = QuantumCircuit(3)   # qubit 0: Alice's message, qubit 1: Alice's twin, qubit 2: Bob's twin

# 1. Alice's secret message: an odd blend with a hidden direction
qc.ry(2.214, 0)          # 20/80 odds
qc.s(0)                  # plus a quarter-turn she wants to send too

# 2. Make a Bell pair (Kata 4!): Alice keeps qubit 1, Bob takes qubit 2 far away
qc.h(1)
qc.cx(1, 2)

# 3. Alice mixes her message into her twin, then "measures" both her qubits
qc.cx(0, 1)
qc.h(0)

# 4. Bob fixes his qubit using Alice's two results
if FIX_X:
    qc.cx(1, 2)          # if Alice's qubit 1 was 1: flip Bob's qubit
if FIX_Z:
    qc.cz(0, 2)          # if Alice's qubit 0 was 1: fix Bob's sign

# 5. Check: undo the message recipe on Bob's qubit. Perfect teleport -> 100% '0'
qc.sdg(2)                # undo the quarter-turn (sdg = S backwards)
qc.ry(-2.214, 2)         # undo the tip

odds = Statevector(qc).probabilities([2])
print(f"FIX_X={FIX_X}, FIX_Z={FIX_Z}: Bob's check says '0' with {odds[0]:.0%}")
alice = Statevector(qc).probabilities_dict([0, 1])
print("Alice's two results:", {str(k): round(float(v), 2) for k, v in alice.items()})