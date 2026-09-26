"""
Portfolio-Optimierung als QUBO – alle drei Verfahren im Vergleich
=================================================================

Erzeugt zufällige Marktdaten, baut das QUBO und löst es nacheinander mit
    a) Brute Force            – exakt, als Referenz  (brute_force.py)
    b) Simulated Annealing    – klassische Heuristik (simulated_annealing.py)
    c) QAOA                   – Quantenalgorithmus   (qaoa.py)

Die Erklärungen zu QUBO, Markowitz-Modell und Strafterm stehen in
portfolio_common.py, die zu den einzelnen Verfahren in deren Skripten.
Jedes Verfahren lässt sich dort auch einzeln und mit eigenen Optionen starten.

Ausführen:
    gym/venv/bin/python portfolio_qubo.py              # Standardlauf (N = 8, K = 4)
    gym/venv/bin/python portfolio_qubo.py --plot       # zusätzlich Kurse + Schaltkreis
    gym/venv/bin/python portfolio_qubo.py --n-assets 12 --k 3
    gym/venv/bin/python portfolio_qubo.py --ibm        # QAOA zusätzlich auf IBM-Hardware
"""

from __future__ import annotations

import time

import numpy as np

from brute_force import energy_rank, solve_brute_force
from portfolio_common import (describe_portfolio, make_parser, plot_prices, print_distribution,
                              qubo_energy, setup_problem)
from qaoa import plot_circuit, run_on_ibm, solve_qaoa
from simulated_annealing import solve_simulated_annealing


def main() -> None:
    parser = make_parser("Portfolio-QUBO: Brute Force, Simulated Annealing und QAOA im Vergleich")
    parser.add_argument("--reps", type=int, default=3, help="QAOA-Schichten p")
    parser.add_argument("--ibm", action="store_true", help="QAOA-Schaltkreis zusätzlich auf IBM-Hardware ausführen")
    parser.add_argument("--backend", type=str, default=None, help="IBM-Backend (Standard: am wenigsten ausgelastet)")
    parser.add_argument("--shots", type=int, default=4000, help="Anzahl Messungen auf der Hardware")
    parser.add_argument("--job-id", type=str, default=None, help="Ergebnis eines früheren IBM-Jobs abholen")
    args = parser.parse_args()
    if args.n_assets > 20:
        parser.error("Für den Vergleich höchstens N = 20 (Brute Force und QAOA-Simulation). "
                     "Größere N mit simulated_annealing.py bzw. brute_force.py einzeln ausprobieren.")

    problem = setup_problem(args, parser)
    Q, constant, k, data = problem.Q, problem.constant, problem.k, problem.data

    def print_rank(x: np.ndarray) -> None:
        """Platz des gewählten Depots in der Brute-Force-Rangliste (1 = optimal)."""
        print(f"  Rang laut Brute Force: Platz {energy_rank(x, Q, constant):,} von {2**problem.n:,} Depots")

    print("\n" + "=" * 78)
    print("Ergebnisse")
    print("=" * 78)

    start = time.time()
    x_bf, e_bf = solve_brute_force(Q, constant)
    end = time.time()
    print(f"Brute Force         E = {e_bf:+.4f}  x = {x_bf}")
    print(f"Time: {end - start:.3f}s")
    print(f"                    {describe_portfolio(x_bf, data)}")

    start = time.time()
    x_sa, e_sa = solve_simulated_annealing(Q, constant, problem.rng)
    end = time.time()
    print(f"Simulated Annealing E = {e_sa:+.4f}  x = {x_sa}")
    print(f"Time: {end - start:.3f}s")
    print(f"                    {describe_portfolio(x_sa, data)}")
    print_rank(x_sa)

    print(f"\nQAOA läuft (p = {args.reps}, {problem.n} Qubits) …")
    start = time.time()
    x_qa, e_qa, info = solve_qaoa(Q, constant, problem.rng, reps=args.reps)
    end = time.time()
    print(f"QAOA                E = {e_qa:+.4f}  x = {x_qa}")
    print(f"Time: {end - start:.3f}s")
    print(f"                    {describe_portfolio(x_qa, data)}")
    print_rank(x_qa)
    gate_counts = ", ".join(f"{g}: {c}" for g, c in info["circuit"].count_ops().items() if g != "barrier")
    print(f"  Schaltkreis: {gate_counts}  (Tiefe {info['circuit_depth']})")
    print(f"  ⟨H⟩ nach Optimierung: {info['expectation']:+.4f}")
    prob_vector = info["prob_vector"]
    p_opt = prob_vector[int(x_bf @ (1 << np.arange(problem.n)))]
    p_feasible = prob_vector[np.bitwise_count(np.arange(2**problem.n)) == k].sum()
    print_distribution(info["probabilities"], x_bf, k, Q, constant, p_opt=p_opt, p_feasible=p_feasible)

    if args.ibm:
        print("\n" + "=" * 78)
        print("QAOA auf IBM-Quantenhardware (Winkel aus der Simulation)")
        print("=" * 78)
        hw_probabilities = run_on_ibm(info["circuit"], args.backend, args.shots, args.job_id)
        x_hw = min((np.array(x) for x in list(hw_probabilities)[:20]),
                   key=lambda x: qubo_energy(x, Q, constant))
        print(f"QAOA (Hardware)     E = {qubo_energy(x_hw, Q, constant):+.4f}  x = {x_hw}")
        print(f"                    {describe_portfolio(x_hw, data)}")
        print_rank(x_hw)
        print_distribution(hw_probabilities, x_bf, k, Q, constant)

    if args.plot:
        import matplotlib.pyplot as plt

        suffix = f"{problem.n}_{k}"
        plot_prices(data, x_bf, ", optimal", f"kurse_{suffix}.png")
        plot_circuit(info["circuit"], args.reps, f"qaoa_circuit_{suffix}.png")
        plt.show()


if __name__ == "__main__":
    main()
