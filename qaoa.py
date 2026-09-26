"""
QAOA – Quantum Approximate Optimization Algorithm für das Portfolio-QUBO
========================================================================

Löst das QUBO mit dem Quantenalgorithmus QAOA:
  * die Winkel γ, β werden in einer schnellen Zustandsvektor-Simulation optimiert,
  * der fertige Schaltkreis kann optional auf echter IBM-Hardware laufen (--ibm).

N Aktien = N Qubits. Die Simulation speichert 2^N komplexe Amplituden:
    N = 16  →  1 MB        schnell
    N = 20  →  16 MB       dauert spürbar (Minuten)
    N = 24  →  256 MB      sehr langsam
Für N ≤ 20 wird zum Vergleich zusätzlich die exakte Lösung per Brute Force
berechnet, damit man sieht, wie wahrscheinlich QAOA das Optimum misst.

Ausführen:
    gym/venv/bin/python qaoa.py
    gym/venv/bin/python qaoa.py --n-assets 6 --k 3 --reps 5
    gym/venv/bin/python qaoa.py --n-assets 4 --k 2 --reps 1 --plot   # kleiner Schaltkreis
    gym/venv/bin/python qaoa.py --ibm                                 # zusätzlich auf IBM-Hardware
    gym/venv/bin/python qaoa.py --ibm --backend ibm_torino --shots 4000
    gym/venv/bin/python qaoa.py --ibm --job-id <ID>                   # Ergebnis abholen

Für --ibm muss der Account einmalig mit QiskitRuntimeService.save_account(...)
gespeichert sein (siehe setup.ipynb).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from brute_force import solve_brute_force
from portfolio_common import (describe_portfolio, make_parser, plot_prices, print_distribution,
                              qubo_energy, qubo_to_ising, setup_problem)


# =============================================================================
# Schaltkreis und schnelle Simulation
# =============================================================================

def build_qaoa_circuit(hamiltonian: SparsePauliOp, reps: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Baut den QAOA-Schaltkreis explizit aus Standardgattern auf.

    Die Winkel bleiben zunächst symbolisch (γ_1..γ_p, β_1..β_p) und werden erst
    nach der Optimierung mit Zahlen belegt. Aufbau:

        H auf allen Qubits                         → |+⟩^⊗N
        p-mal:
          Kosten-Layer exp(−i·γ·H_C), Term für Term aus dem SparsePauliOp:
            h_i · Z_i        →  RZ(2γ·h_i)      auf Qubit i
            J_ij · Z_i Z_j   →  RZZ(2γ·J_ij)    auf Qubits i, j
          Mixer-Layer exp(−i·β·Σ X_i):
            RX(2β)                              auf jedem Qubit

    Der Faktor 2 kommt daher, dass Qiskit die Rotationen als RZ(θ) = exp(−i·θ/2·Z)
    definiert. Weil alle Terme von H_C nur aus Z bestehen, vertauschen sie
    miteinander – deshalb ist exp(−iγH_C) exakt das Produkt der Einzelgatter
    (keine Trotter-Näherung nötig).
    """
    n = hamiltonian.num_qubits
    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)

    qc = QuantumCircuit(n, name="QAOA")
    qc.h(range(n))
    for layer in range(reps):
        qc.barrier()
        for label, qubits, coeff in hamiltonian.to_sparse_list():
            angle = 2 * gammas[layer] * float(np.real(coeff))
            if label == "Z":
                qc.rz(angle, qubits[0])
            elif label == "ZZ":
                qc.rzz(angle, qubits[0], qubits[1])
        qc.barrier()
        qc.rx(2 * betas[layer], range(n))
    return qc, gammas, betas


def ising_diagonal(hamiltonian: SparsePauliOp) -> np.ndarray:
    """
    Energie jedes Basiszustands |k⟩ unter dem (nur aus Z bestehenden) H_C.

    Z_i hat auf Bit i den Eigenwert z_i = 1 − 2·x_i, also ist die Energie von |k⟩
    einfach Σ coeff · Π z_i über alle Terme. Bit i von k gehört zu Qubit i
    (Qiskit-Konvention, siehe Sanity-Check in main).
    """
    n = hamiltonian.num_qubits
    indices = np.arange(2**n)
    # z[i] = Eigenwerte von Z_i für alle 2^N Basiszustände (int8 spart Speicher)
    z = [(1 - 2 * ((indices >> i) & 1)).astype(np.int8) for i in range(n)]
    diag = np.zeros(2**n)
    for _, qubits, coeff in hamiltonian.to_sparse_list():
        term = z[qubits[0]] if len(qubits) == 1 else z[qubits[0]] * z[qubits[1]]
        diag += float(np.real(coeff)) * term
    return diag


def simulate_qaoa(cost_diag: np.ndarray, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """
    Schnelle Zustandsvektor-Simulation genau des Schaltkreises aus build_qaoa_circuit.

    Warum nicht einfach Statevector(circuit)? Das funktioniert (und wird am Ende
    zur Kontrolle auch gemacht), ist aber für die Optimierungsschleife mit
    tausenden Aufrufen unnötig langsam. Hier nutzen wir die Struktur aus:

      * Kosten-Layer: H_C ist diagonal → exp(−iγH_C) multipliziert jede
        Amplitude nur mit einer Phase:  ψ_k ← exp(−iγ·E_k) · ψ_k
      * Mixer-Layer: RX(2β) = [[cos β, −i sin β], [−i sin β, cos β]] wirkt auf
        jedes Qubit einzeln. Mit reshape wird Qubit q zu einer eigenen Achse,
        auf der die 2×2-Matrix angewendet wird.

    Eine Auswertung kostet so nur ~p·N Vektoroperationen der Länge 2^N.
    """
    n = int(np.log2(len(cost_diag)))
    psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)          # H^⊗N |0…0⟩
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * cost_diag)                  # Kosten-Layer
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(n):                                      # Mixer-Layer
            view = psi.reshape(-1, 2, 2**q)                     # Achse 1 = Qubit q
            a0, a1 = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = c * a0 + s * a1
            view[:, 1, :] = s * a0 + c * a1
    return psi


# =============================================================================
# QAOA-Optimierung
# =============================================================================

def solve_qaoa(Q: np.ndarray, constant: float, rng: np.random.Generator,
               reps: int = 3, n_restarts: int = 5) -> tuple[np.ndarray, float, dict]:
    """
    QAOA (Farhi, Goldstone, Gutmann, 2014) ist ein hybrider Quanten-/klassischer
    Algorithmus für genau solche Ising-/QUBO-Probleme.

    Idee – eine diskretisierte Version des adiabatischen Quantencomputings:

      1. Start in der gleichmäßigen Überlagerung aller 2^N Bitstrings:
             |+⟩^⊗N  = H^⊗N |0…0⟩
         Jedes mögliche Depot ist gleich wahrscheinlich.

      2. Wende abwechselnd p-mal ("reps") zwei Operatoren an:

           Kosten-Layer   U_C(γ) = exp(−i·γ·H_C)
               Gibt jedem Bitstring eine Phase proportional zu seiner Energie.
               Schaltkreis: RZ-Gatter für h_i·Z_i, RZZ-Gatter für J_ij·Z_i Z_j.

           Mixer-Layer    U_M(β) = exp(−i·β·Σ_i X_i)
               RX-Rotationen auf allen Qubits. Mischt die Amplituden, sodass
               die Phasen aus dem Kosten-Layer zu Interferenz führen:
               gute Lösungen werden verstärkt, schlechte ausgelöscht.

         Der Zustand hängt also von 2p Winkeln (γ_1, β_1, …, γ_p, β_p) ab.

      3. Ein klassischer Optimierer (hier COBYLA) sucht Winkel, die den
         Erwartungswert  ⟨ψ(γ,β)| H_C |ψ(γ,β)⟩  minimieren.
         Auf echter Hardware würde man diesen Wert durch wiederholtes Messen
         schätzen; hier rechnen wir exakt mit dem Zustandsvektor
         (schnelle NumPy-Simulation, siehe simulate_qaoa).

      4. Mit den besten Winkeln wird der Zustand präpariert und gemessen.
         Bitstrings mit niedriger Energie treten nun mit hoher
         Wahrscheinlichkeit auf – der wahrscheinlichste gültige ist unser Depot.

    Für p → ∞ wird QAOA exakt (adiabatischer Grenzfall). Schon kleine p
    heben die Wahrscheinlichkeit guter Lösungen aber deutlich über den
    Zufallswert 1/2^N.

    Hinweis: N Aktien = N Qubits. Die Statevector-Simulation braucht 2^N
    komplexe Zahlen – bis ca. 20 Qubits ist das auf einem Laptop machbar.
    """
    hamiltonian, offset = qubo_to_ising(Q, constant)
    n = hamiltonian.num_qubits

    # Skalieren, damit die Winkel γ in einem "vernünftigen" Bereich liegen.
    # (Ändert den Grundzustand nicht, nur die Energieeinheit.)
    scale = np.abs(hamiltonian.coeffs).max()
    h_scaled = (hamiltonian / scale).simplify()

    circuit, gamma_params, beta_params = build_qaoa_circuit(h_scaled, reps)
    cost_diag = ising_diagonal(h_scaled)

    # Parameter-Vektor für den Optimierer: [γ_1..γ_p, β_1..β_p]
    def expectation(params: np.ndarray) -> float:
        psi = simulate_qaoa(cost_diag, params[:reps], params[reps:])
        return float(np.abs(psi) ** 2 @ cost_diag)

    best_result = None
    for _ in range(n_restarts):               # mehrere Starts gegen lokale Minima
        x0 = rng.uniform(0, np.pi, 2 * reps)
        result = minimize(expectation, x0, method="COBYLA", options={"maxiter": 500})
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    gammas, betas = best_result.x[:reps], best_result.x[reps:]
    psi = simulate_qaoa(cost_diag, gammas, betas)

    bound_circuit = circuit.assign_parameters(
        dict(zip(gamma_params, gammas)) | dict(zip(beta_params, betas))
    )
    # Kontrolle: Der echte Qiskit-Schaltkreis mit den optimierten Winkeln muss
    # exakt denselben Zustand erzeugen wie unsere schnelle Simulation.
    # (Nur bis 16 Qubits – darüber dauert die Qiskit-Simulation zu lange.)
    if n <= 16:
        assert np.allclose(Statevector(bound_circuit).data, psi), "Schaltkreis ≠ Simulation"

    # Messwahrscheinlichkeiten |ψ_k|²; Bit i von k = Aktie i.
    # Als Liste werden nur die wahrscheinlichsten Zustände gespeichert (bei
    # N = 20 wären es sonst ~1 Mio. Einträge); die vollständige Verteilung
    # steht in info["prob_vector"].
    probabilities = np.abs(psi) ** 2
    order = np.argsort(-probabilities)[:1000]
    ranked = [(np.array([(k >> i) & 1 for i in range(n)]), probabilities[k]) for k in order]
    best_x = min((x for x, p in ranked[:20]), key=lambda x: qubo_energy(x, Q, constant))

    info = {
        "probabilities": {tuple(x): p for x, p in ranked},
        "prob_vector": probabilities,
        "expectation": best_result.fun * scale + offset,
        "circuit": bound_circuit,
        "circuit_depth": bound_circuit.depth(lambda ins: ins.operation.name != "barrier"),
    }
    return best_x, qubo_energy(best_x, Q, constant), info


# =============================================================================
# QAOA auf echter IBM-Quantenhardware
# =============================================================================

def run_on_ibm(circuit: QuantumCircuit, backend_name: str | None, shots: int,
               job_id: str | None = None) -> dict[tuple, float]:
    """
    Führt den fertigen QAOA-Schaltkreis (mit den lokal optimierten Winkeln) auf
    einem echten IBM-Quantencomputer aus und gibt die gemessene Verteilung zurück.

    Warum nur der fertige Schaltkreis und nicht die ganze Optimierung?
    Die Winkelsuche braucht tausende Auswertungen von ⟨H⟩. Auf Hardware wäre
    jede davon ein Job mit tausenden Shots, dazu Warteschlange und Rauschen –
    das sprengt jedes Kontingent (Open Plan: ca. 10 Minuten pro Monat). Deshalb
    der übliche Weg: Winkel klassisch finden, Hardware nur zum Sampeln nutzen.
    So sieht man direkt, wie viel vom idealen Ergebnis auf echter Hardware übrig
    bleibt.

    Ablauf:
      1. Verbindung über den gespeicherten Account (QiskitRuntimeService)
      2. Backend wählen: angegebenes Gerät oder das am wenigsten ausgelastete
      3. Transpilieren: Der Schaltkreis wird in die nativen Gatter des Chips
         übersetzt (z. B. CZ + RZ + SX statt RZZ + RX) und auf dessen
         Kopplungsgraph abgebildet. Da der Chip nicht jedes Qubit mit jedem
         verbindet, unser QUBO aber alle Paare koppelt, kommen SWAP-Gatter dazu.
      4. Mit SamplerV2 ausführen und die Zählraten in Wahrscheinlichkeiten umrechnen

    Mit job_id wird stattdessen das Ergebnis eines bereits abgeschickten Jobs
    abgeholt (praktisch, wenn die Warteschlange lang ist und das Skript
    zwischendurch beendet wurde). Gleiche Argumente (v. a. --seed) verwenden!
    """
    # Nur importieren, wenn wirklich gebraucht – der Rest läuft auch ohne Account
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()

    if job_id:
        print(f"  Hole Ergebnis von Job {job_id} …")
        job = service.job(job_id)
    else:
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False,
                                         min_num_qubits=circuit.num_qubits)
        print(f"  Backend: {backend.name} ({backend.num_qubits} Qubits, "
              f"{backend.status().pending_jobs} Jobs in der Warteschlange)")

        # Transpilieren: optimization_level=3 sucht am gründlichsten nach einer
        # guten Qubit-Zuordnung und möglichst wenigen SWAPs.
        measured = circuit.measure_all(inplace=False)
        pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
        isa_circuit = pass_manager.run(measured)

        two_qubit_before = sum(1 for ins in circuit.data if ins.operation.num_qubits == 2)
        two_qubit_after = isa_circuit.num_nonlocal_gates()
        print(f"  Transpiliert: {two_qubit_before} → {two_qubit_after} Zwei-Qubit-Gatter, "
              f"Tiefe {isa_circuit.depth()}  (Gatter: {dict(isa_circuit.count_ops())})")

        sampler = SamplerV2(mode=backend)
        # Dynamical Decoupling: füllt Leerlaufzeiten der Qubits mit Pulsfolgen,
        # die sich gegenseitig aufheben – reduziert Dekohärenz, kostet nichts extra.
        sampler.options.dynamical_decoupling.enable = True

        job = sampler.run([isa_circuit], shots=shots)
        print(f"  Job abgeschickt: {job.job_id()}")
        print("  Warte auf Ergebnis (je nach Warteschlange Minuten bis Stunden) …")
        print(f"  Abbrechen ist ok – später abholen mit: --ibm --job-id {job.job_id()}")

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    # Qiskit-Bitstrings haben Qubit 0 ganz rechts → umdrehen, damit x[i] = Aktie i
    probabilities = {
        tuple(int(b) for b in reversed(bitstring)): c / total
        for bitstring, c in counts.items()
    }
    return dict(sorted(probabilities.items(), key=lambda item: -item[1]))


def plot_circuit(circuit: QuantumCircuit, reps: int) -> None:
    """
    QAOA-Schaltkreis mit den optimierten Winkeln (plus Messung am Ende,
    wie er auf echter Hardware laufen würde). Barriers trennen die Layer.
    Qiskit legt die Figur so groß an, dass jedes Gatter lesbar ist (bei
    8 Qubits, p = 3 ca. 35 × 41 Zoll). Ein Plot-Fenster skaliert das auf
    Bildschirmgröße herunter – dann verschwinden die dünnen Linien. Deshalb
    zusätzlich in voller Größe als PNG speichern (zum Reinzoomen).
    """
    circuit_fig = circuit.measure_all(inplace=False).draw("mpl", fold=40)
    circuit_fig.suptitle(f"QAOA-Schaltkreis (p = {reps}, optimierte Winkel)", fontsize=24)
    circuit_fig.savefig("qaoa_circuit.png", dpi=100, bbox_inches="tight")
    print("\nSchaltkreis gespeichert: qaoa_circuit.png")


def main() -> None:
    parser = make_parser("Portfolio-QUBO per QAOA lösen")
    parser.add_argument("--reps", type=int, default=3, help="QAOA-Schichten p")
    parser.add_argument("--restarts", type=int, default=5, help="Neustarts der Winkeloptimierung")
    parser.add_argument("--ibm", action="store_true", help="QAOA-Schaltkreis zusätzlich auf IBM-Hardware ausführen")
    parser.add_argument("--backend", type=str, default=None, help="IBM-Backend (Standard: am wenigsten ausgelastet)")
    parser.add_argument("--shots", type=int, default=4000, help="Anzahl Messungen auf der Hardware")
    parser.add_argument("--job-id", type=str, default=None, help="Ergebnis eines früheren IBM-Jobs abholen")
    args = parser.parse_args()
    if args.n_assets > 24:
        parser.error("Mehr als 24 Qubits sind für die Zustandsvektor-Simulation zu groß")

    problem = setup_problem(args, parser)
    Q, constant, k = problem.Q, problem.constant, problem.k

    # Sanity-Check: Ising-Form liefert für jeden Bitstring dieselbe Energie
    hamiltonian, offset = qubo_to_ising(Q, constant)
    diag = ising_diagonal(hamiltonian) + offset
    for idx in problem.rng.integers(0, 2**problem.n, 5):
        x = np.array([(idx >> i) & 1 for i in range(problem.n)])
        assert np.isclose(diag[idx], qubo_energy(x, Q, constant)), "QUBO ≠ Ising"

    # Exakte Referenz, um die QAOA-Verteilung einordnen zu können
    x_ref = None
    if problem.n <= 20:
        x_ref, e_ref = solve_brute_force(Q, constant)
        print(f"\nReferenz (Brute Force): E = {e_ref:+.4f}  x = {x_ref}")

    print("\n" + "=" * 78)
    print(f"QAOA läuft (p = {args.reps}, {problem.n} Qubits) …")
    print("=" * 78)
    start = time.time()
    x_qa, e_qa, info = solve_qaoa(Q, constant, problem.rng, reps=args.reps, n_restarts=args.restarts)
    end = time.time()
    print(f"QAOA                E = {e_qa:+.4f}  x = {x_qa}")
    print(f"Time: {end - start:.3f}s")
    print(f"                    {describe_portfolio(x_qa, problem.data)}")
    gate_counts = ", ".join(f"{g}: {c}" for g, c in info["circuit"].count_ops().items() if g != "barrier")
    print(f"  Schaltkreis: {gate_counts}  (Tiefe {info['circuit_depth']})")
    print(f"  ⟨H⟩ nach Optimierung: {info['expectation']:+.4f}")
    prob_vector = info["prob_vector"]
    p_opt = prob_vector[int(x_ref @ (1 << np.arange(problem.n)))] if x_ref is not None else None
    p_feasible = prob_vector[np.bitwise_count(np.arange(2**problem.n)) == k].sum()
    print_distribution(info["probabilities"], x_ref, k, Q, constant, p_opt=p_opt, p_feasible=p_feasible)

    # --- Optional: echte IBM-Hardware -----------------------------------------
    if args.ibm:
        print("\n" + "=" * 78)
        print("QAOA auf IBM-Quantenhardware (Winkel aus der Simulation)")
        print("=" * 78)
        hw_probabilities = run_on_ibm(info["circuit"], args.backend, args.shots, args.job_id)

        # Wie in der Simulation: unter den 20 häufigsten Messungen das beste Depot
        x_hw = min((np.array(x) for x in list(hw_probabilities)[:20]),
                   key=lambda x: qubo_energy(x, Q, constant))
        print(f"QAOA (Hardware)     E = {qubo_energy(x_hw, Q, constant):+.4f}  x = {x_hw}")
        print(f"                    {describe_portfolio(x_hw, problem.data)}")
        print_distribution(hw_probabilities, x_ref, k, Q, constant)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_prices(problem.data, x_qa, ", QAOA")
        plot_circuit(info["circuit"], args.reps)
        plt.show()


if __name__ == "__main__":
    main()
