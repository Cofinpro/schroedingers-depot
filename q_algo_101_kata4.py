from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt



qc = QuantumCircuit(2)   # one qubit, starts at 0

#1. qc.x(0) only. Predict the bitstring (mind the Qiskit gotcha: it writes qubit 0 on the right. So '01' means qubit 0 is 1 and qubit 1 is 0.).
#qc.x(0)

#2. qc.x(0) then qc.cx(0, 1). Predict first.
#qc.x(0), qc.cx(0, 1)

#3. qc.h(0) then qc.cx(0, 1). This is the famous Bell state. Also try Statevector(qc).sample_counts(10) to "measure" it 10 times.
qc.h(0), qc.cx(0, 1)

counts = Statevector(qc).sample_counts(10) #Statevector(qc).sample_counts(1000)  # measure 1000 times
print(counts)
#print(Statevector(qc))
print(qc.draw())
probs = Statevector(qc).probabilities_dict()
print({str(k): round(float(v), 3) for k, v in probs.items()})

qc.draw("mpl")
plt.show()