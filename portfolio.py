import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# --- 1. The problem (made-up numbers) ---------------------------------
STOCKS = ["Apple", "Bank", "Coal"]
returns = np.array([0.30, 0.20, 0.25])        # expected yearly return
risk = np.array([[0.10, 0.02, 0.08],          # how much they wobble together
                 [0.02, 0.05, 0.01],          # (covariance matrix)
                 [0.08, 0.01, 0.40]])
RISK_AVERSION = 0.5   # how much you hate risk
BUDGET = 2            # buy exactly 2 stocks
PENALTY = 0.5         # punishment for breaking the budget

N = len(STOCKS)

def bits_of(i):
    """Portfolio number i -> list of 0/1 (qubit 0 = first stock)."""
    return np.array([(i >> k) & 1 for k in range(N)])

def cost(x):
    """Lower is better: risk minus return, plus a penalty for the wrong number of stocks."""
    return RISK_AVERSION * x @ risk @ x - returns @ x + PENALTY * (x.sum() - BUDGET) ** 2

costs = np.array([cost(bits_of(i)) for i in range(2 ** N)])  # one cost per portfolio

# --- 2. The quantum algorithm: QAOA -----------------------------------
LAYERS = 2

def qaoa_circuit(angles):
    gammas, betas = angles[:LAYERS], angles[LAYERS:]
    qc = QuantumCircuit(N)
    qc.h(range(N))                                  # blend all 8 portfolios
    for gamma, beta in zip(gammas, betas):
        qc.append(DiagonalGate(list(np.exp(-1j * gamma * costs))), range(N))  # cost layer: turn ~ cost
        qc.rx(2 * beta, range(N))                   # mixer layer: let the weights flow
    return qc

def average_cost(angles):
    probs = Statevector(qaoa_circuit(angles)).probabilities()
    return probs @ costs

# the classical computer tunes the angles, the quantum circuit scores them
rng = np.random.default_rng(seed=7)
best = min((minimize(average_cost, rng.uniform(0, 2 * np.pi, 2 * LAYERS), method="COBYLA")
            for _ in range(10)), key=lambda r: r.fun)

# --- 3. Results -------------------------------------------------------
q_circuit=qaoa_circuit(best.x)
probs = Statevector(q_circuit).probabilities()
print("portfolio          cost    chance")
for i in np.argsort(-probs)[:4]:  # top 4 answers
    picked = [s for s, b in zip(STOCKS, bits_of(i)) if b] or ["nothing"]
    print(f"{'+'.join(picked):16} {costs[i]:7.3f}   {probs[i]:6.1%}")
print("true best (checked by brute force):",
      "+".join(s for s, b in zip(STOCKS, bits_of(np.argmin(costs))) if b))


# print(Statevector(q_circuit))
# print(q_circuit.draw())
# probs_dic = Statevector(q_circuit).probabilities_dict()
# print({str(k): round(float(v), 3) for k, v in probs_dic.items()})
# Statevector(q_circuit).draw('bloch')
# q_circuit.draw("mpl")
# plt.show()