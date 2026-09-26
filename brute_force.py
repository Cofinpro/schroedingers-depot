"""
Brute Force – exakte Lösung des Portfolio-QUBO
==============================================

Probiert alle 2^N möglichen Depots durch und nimmt das mit der kleinsten
QUBO-Energie. Das Ergebnis ist garantiert optimal und dient als Referenz für
Simulated Annealing und QAOA.

Laufzeit: verdoppelt sich mit jeder zusätzlichen Aktie.
    N = 20  →  ~1 Mio. Depots       (Sekundenbruchteile)
    N = 25  →  ~34 Mio. Depots      (einige Sekunden)
    N = 30  →  ~1 Mrd. Depots       (Minuten)
    N = 50  →  ~10^15 Depots        (unmöglich)

Ausführen:
    gym/venv/bin/python brute_force.py
    gym/venv/bin/python brute_force.py --n-assets 16 --k 5
    gym/venv/bin/python brute_force.py --n-assets 24 --k 8 --plot
"""

from __future__ import annotations

import time

import numpy as np

from portfolio_common import describe_portfolio, make_parser, plot_prices, setup_problem


def solve_brute_force(Q: np.ndarray, constant: float,
                      chunk_bits: int = 20) -> tuple[np.ndarray, float]:
    """
    Probiert alle 2^N Bitstrings durch. Garantiert optimal, aber exponentiell:
    N = 20 → ~1 Mio. Kombinationen, N = 50 → ~10^15. Genau deshalb braucht man
    für große Probleme Heuristiken oder (hoffentlich irgendwann) Quantenhardware.

    Umsetzung: Statt jedes Depot einzeln in einer Python-Schleife auszuwerten,
    werden jeweils 2^chunk_bits Depots auf einmal als Matrix X (eine Zeile pro
    Depot) verarbeitet. Die Energie aller Zeilen ist dann

        E = Σ_ij X_ki Q_ij X_kj + c   =   ((X @ Q) * X).sum(axis=1) + c

    Das ist mathematisch dasselbe wie xᵀQx für jede Zeile, aber NumPy rechnet
    es in optimiertem C-Code – um Größenordnungen schneller. Der Speicher
    bleibt durch die Blöcke begrenzt, egal wie groß N ist.

    Depot Nummer idx entspricht dem Bitstring mit x_i = Bit i von idx
    (dieselbe Konvention wie bei den Qubits in Qiskit).
    """
    n = Q.shape[0]
    bit_positions = np.arange(n)
    best_idx, best_e = -1, np.inf

    for start, energies in iter_energies(Q, constant, chunk_bits):
        i = int(np.argmin(energies))
        if energies[i] < best_e:
            best_idx, best_e = start + i, float(energies[i])

    best_x = (best_idx >> bit_positions) & 1
    return best_x, best_e


def iter_energies(Q: np.ndarray, constant: float, chunk_bits: int = 20):
    """
    Liefert die QUBO-Energien aller 2^N Depots blockweise als (Startindex, Energien).
    Energien[j] gehört zu Depot Nummer Startindex + j.
    """
    n = Q.shape[0]
    chunk = 2 ** min(n, chunk_bits)
    bit_positions = np.arange(n)
    for start in range(0, 2**n, chunk):
        idx = np.arange(start, start + chunk, dtype=np.int64)
        X = ((idx[:, None] >> bit_positions) & 1).astype(np.float64)
        yield start, ((X @ Q) * X).sum(axis=1) + constant


def energy_rank(x: np.ndarray, Q: np.ndarray, constant: float, tol: float = 1e-9) -> int:
    """
    Platz des Depots x in der Rangliste aller 2^N Depots, sortiert nach Energie
    (Platz 1 = optimal). Gezählt wird, wie viele Depots eine echt kleinere
    Energie haben; Depots mit gleicher Energie teilen sich denselben Platz.
    """
    e = float(x @ Q @ x + constant)
    better = sum(int((energies < e - tol).sum()) for _, energies in iter_energies(Q, constant))
    return better + 1


def main() -> None:
    parser = make_parser("Portfolio-QUBO per Brute Force lösen")
    args = parser.parse_args()
    if args.n_assets > 30:
        parser.error(f"2^{args.n_assets} Depots sind zu viele für Brute Force (max. N = 30)")

    problem = setup_problem(args, parser)
    if problem.n > 26:
        print(f"\nAchtung: 2^{problem.n} Depots – das kann einige Minuten dauern.")

    start = time.time()
    x, e = solve_brute_force(problem.Q, problem.constant)
    end = time.time()

    print("\n" + "=" * 78)
    print(f"Brute Force         E = {e:+.4f}  x = {x}")
    print(f"Time: {end - start:.3f}s")
    print(f"                    {describe_portfolio(x, problem.data)}")

    if args.plot:
        import matplotlib.pyplot as plt

        plot_prices(problem.data, x, ", Brute Force", f"kurse_brute_force_{problem.n}_{problem.k}.png")
        plt.show()


if __name__ == "__main__":
    main()
