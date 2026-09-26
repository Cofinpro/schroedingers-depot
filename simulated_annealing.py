"""
Simulated Annealing – klassische Heuristik für das Portfolio-QUBO
=================================================================

Findet gute (meist optimale) Depots, ohne alle 2^N Möglichkeiten auszuprobieren.
Die Laufzeit wächst nur linear mit N und der Zahl der Sweeps – funktioniert
also auch für N = 50, 100 oder mehr, wo Brute Force aussteigt. Eine Garantie
für das Optimum gibt es dafür nicht; mehr Sweeps und Neustarts erhöhen die
Trefferquote.

Ausführen:
    gym/venv/bin/python simulated_annealing.py
    gym/venv/bin/python simulated_annealing.py --n-assets 40 --k 10
    gym/venv/bin/python simulated_annealing.py --n-assets 40 --k 10 --sweeps 5000 --restarts 10
"""

from __future__ import annotations

import time

import numpy as np

from portfolio_common import describe_portfolio, make_parser, plot_prices, qubo_energy, setup_problem


def solve_simulated_annealing(Q: np.ndarray, constant: float, rng: np.random.Generator,
                              n_sweeps: int = 2000, t_start: float | None = None,
                              t_end: float = 1e-3) -> tuple[np.ndarray, float]:
    """
    Simulated Annealing imitiert das langsame Abkühlen eines Metalls:

      * Starte mit einer zufälligen Lösung und einer hohen "Temperatur" T.
      * Schlage vor, ein einzelnes Bit x_i umzudrehen (0 ↔ 1).
      * Wird die Energie kleiner  → immer annehmen.
        Wird sie größer um ΔE     → nur mit Wahrscheinlichkeit exp(−ΔE / T) annehmen.
      * Senke T schrittweise ab.

    Bei hohem T springt der Algorithmus fast frei umher (Exploration), bei
    niedrigem T nimmt er nur noch Verbesserungen an (Exploitation). So kann er
    aus lokalen Minima wieder herausklettern.

    Effizienter Trick: Beim Umdrehen von Bit i ändert sich die Energie um
        ΔE = (1 − 2x_i) · (Q_ii + 2·Σ_{j≠i} Q_ij x_j)
    Man muss also nicht jedes Mal xᵀQx komplett neu ausrechnen.

    (Quantenannealer wie D-Wave machen konzeptionell etwas Ähnliches, nutzen
    aber Quantentunneln statt thermischer Sprünge.)
    """
    n = Q.shape[0]
    if t_start is None:
        t_start = np.abs(Q).max()
    temperatures = np.geomspace(t_start, t_end, n_sweeps)

    x = rng.integers(0, 2, n)
    energy = qubo_energy(x, Q, constant)
    best_x, best_e = x.copy(), energy

    for t in temperatures:
        for i in rng.permutation(n):          # ein "Sweep" = jedes Bit einmal versuchen
            local_field = Q[i, i] + 2 * (Q[i] @ x - Q[i, i] * x[i])
            delta = (1 - 2 * x[i]) * local_field
            if delta < 0 or rng.random() < np.exp(-delta / t):
                x[i] ^= 1
                energy += delta
                if energy < best_e:
                    best_x, best_e = x.copy(), energy
    return best_x, best_e


def main() -> None:
    parser = make_parser("Portfolio-QUBO per Simulated Annealing lösen")
    parser.add_argument("--sweeps", type=int, default=2000, help="Abkühlschritte pro Lauf")
    parser.add_argument("--restarts", type=int, default=5,
                        help="unabhängige Läufe; das beste Ergebnis zählt")
    args = parser.parse_args()

    problem = setup_problem(args, parser)

    start = time.time()
    results = [solve_simulated_annealing(problem.Q, problem.constant, problem.rng, n_sweeps=args.sweeps)
               for _ in range(args.restarts)]
    end = time.time()
    x, e = min(results, key=lambda r: r[1])
    n_hits = sum(np.isclose(r[1], e) for r in results)

    print("\n" + "=" * 78)
    print(f"Simulated Annealing E = {e:+.4f}  x = {x}")
    print(f"Time: {end - start:.3f}s  ({args.restarts} Läufe à {args.sweeps} Sweeps, "
          f"{n_hits}× dasselbe beste Ergebnis gefunden)")
    print(f"                    {describe_portfolio(x, problem.data)}")
    if sum(x) != problem.k:
        print(f"  Achtung: {sum(x)} statt K = {problem.k} Aktien – mehr Sweeps versuchen.")

    if args.plot:
        import matplotlib.pyplot as plt

        plot_prices(problem.data, x, ", Simulated Annealing")
        plt.show()


if __name__ == "__main__":
    main()
