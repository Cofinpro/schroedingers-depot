from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

SECRET = "101"
ROUNDS = 2
SHOTS = 1000
QUBITS = [0, 1, 2]

def flip_zeros(qc, secret):
    for i, bit in enumerate(reversed(secret)):
        if bit == "0":
            qc.x(i)

def oracle(qc, secret):
    flip_zeros(qc, secret)
    qc.ccz(0, 1, 2)
    flip_zeros(qc, secret)

def diffuser(qc):
    qc.h(QUBITS)
    qc.x(QUBITS)
    qc.ccz(0, 1, 2)
    qc.x(QUBITS)
    qc.h(QUBITS)

qc = QuantumCircuit(3)
qc.h(QUBITS)
for _ in range(ROUNDS):
    oracle(qc, SECRET)
    diffuser(qc)
qc.measure_all()  # real hardware needs an explicit measurement

perfect = AerSimulator()                          # no noise
noisy = AerSimulator.from_backend(FakeBrisbane())  # copy of a real IBM chip, errors included

for name, sim in [("perfect", perfect), ("noisy", noisy)]:
    chip_circuit = transpile(qc, sim)  # translate into the gates this machine understands
    counts = sim.run(chip_circuit, shots=SHOTS).result().get_counts()
    hits = counts.get(SECRET, 0)
    print(f"{name:8} found {SECRET} in {hits}/{SHOTS} runs ({hits / SHOTS:.1%}), "
          f"circuit size: {chip_circuit.size()} gates")