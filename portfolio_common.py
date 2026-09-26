"""
Portfolio-Optimierung als QUBO – gemeinsamer Teil aller Skripte
===================================================================

Diese Skriptsammlung zeigt den kompletten Weg von (künstlichen) Kursdaten bis zu einem
optimierten Depot, das mit einem QUBO-Modell ausgewählt wird:

    1. Zufällige Aktienkurse erzeugen (korrelierte geometrische Brownsche Bewegung)
    2. Aus den Kursen erwartete Renditen  μ  und die Kovarianzmatrix  Σ  schätzen
    3. Das Auswahlproblem "Welche K von N Aktien kaufe ich?" als QUBO formulieren
    4. Das QUBO auf drei Arten lösen:
         a) Brute Force            – exakt, als Referenz (nur für kleine N möglich)
         b) Simulated Annealing    – klassische Heuristik
         c) QAOA                   – Quantenalgorithmus (simuliert mit Qiskit)

---------------------------------------------------------------------------
Was ist ein QUBO?
---------------------------------------------------------------------------
QUBO steht für "Quadratic Unconstrained Binary Optimization". Gesucht ist ein
Vektor aus Nullen und Einsen  x = (x_0, ..., x_{N-1}),  x_i ∈ {0, 1},  der die
quadratische Funktion

        E(x) = xᵀ Q x  =  Σ_i Q_ii x_i  +  Σ_{i≠j} Q_ij x_i x_j

minimiert. (Weil x_i² = x_i gilt, stecken die linearen Terme auf der Diagonalen.)

  * "Quadratic"     – nur Terme bis zur Ordnung 2 (x_i · x_j)
  * "Unconstrained" – keine Nebenbedingungen; diese müssen als Strafterme
                      (Penalty) in Q eingebaut werden
  * "Binary"        – jede Variable ist 0 oder 1

QUBOs sind interessant, weil sie 1:1 in ein Ising-Modell übersetzt werden
können – und genau diese Form können Quantenannealer (D-Wave) und
gatterbasierte Quantencomputer (über QAOA) direkt verarbeiten.

---------------------------------------------------------------------------
Das Portfolio-Problem (nach Markowitz)
---------------------------------------------------------------------------
x_i = 1 bedeutet: "Aktie i kommt ins Depot", x_i = 0: "nicht".
Wir wollen

    minimiere   q · xᵀ Σ x      (Risiko   = Varianz des Depots)
              −     μᵀ x        (Rendite  – mit Minus, weil wir minimieren)
    unter       Σ_i x_i = K     (genau K Aktien auswählen, gleich gewichtet)

q ist die Risikoaversion: q groß → vorsichtiges Depot, q klein → renditehungrig.

Die Nebenbedingung wird zur Strafe  P · (Σ_i x_i − K)²  – sie ist genau dann 0,
wenn exakt K Aktien gewählt sind, sonst positiv. Mit hinreichend großem P lohnt
es sich für den Optimierer nie, die Bedingung zu verletzen.

---------------------------------------------------------------------------
Aufteilung der Skripte
---------------------------------------------------------------------------
    portfolio_common.py     Daten, QUBO, Ising-Umrechnung, Ausgabe (dieses Modul)
    brute_force.py          exakte Lösung durch Ausprobieren aller 2^N Depots
    simulated_annealing.py  klassische Heuristik
    qaoa.py                 Quantenalgorithmus (Simulation, optional IBM-Hardware)
    portfolio_qubo.py       alle drei nacheinander zum direkten Vergleich

Alle Skripte haben dieselben Grundargumente (--n-assets, --k, --q, --days,
--seed). Mit gleichem --seed erzeugen sie identische Marktdaten, die
Ergebnisse sind also direkt vergleichbar.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# =============================================================================
# 1. Zufällige Aktiendaten erzeugen
# =============================================================================

TICKER_POOL = [
    "QBIT", "SCHR", "ENTG", "SUPR", "HADM", "PAUL", "BLOC", "DIRC",
    "HEIS", "BORN", "PLNK", "FEYN", "DIRA", "NOET", "BOSE", "FERM",
]


def make_tickers(n_assets: int) -> list[str]:
    """Fantasie-Ticker; für mehr als 16 Aktien wird durchnummeriert."""
    extra = [f"AKT{i:02d}" for i in range(len(TICKER_POOL), n_assets)]
    return (TICKER_POOL + extra)[:n_assets]


@dataclass
class MarketData:
    tickers: list[str]
    prices: np.ndarray        # Form (Tage + 1, N)
    mu: np.ndarray            # erwartete Jahresrendite je Aktie, Form (N,)
    sigma: np.ndarray         # Kovarianzmatrix der Jahresrenditen, Form (N, N)


def generate_market_data(n_assets: int, n_days: int, rng: np.random.Generator) -> MarketData:
    """
    Simuliert Tageskurse für n_assets Aktien über n_days Handelstage.

    Modell: Geometrische Brownsche Bewegung (GBM) mit korrelierten Schocks.
    Die Korrelation entsteht über ein einfaches Faktormodell:

        r_t = Drift + B · f_t + ε_t

      * f_t : 2 gemeinsame Marktfaktoren (z. B. "Gesamtmarkt", "Branche")
      * B   : wie stark jede Aktie auf die Faktoren reagiert (Faktorladungen)
      * ε_t : aktienspezifisches Rauschen

    Dadurch bewegen sich manche Aktien ähnlich – genau das macht Diversifikation
    (und damit die Kovarianzmatrix im QUBO) überhaupt erst interessant.
    """
    tickers = make_tickers(n_assets)
    trading_days = 252

    # "Wahre" Parameter, die wir später nur noch aus den Daten schätzen
    annual_drift = rng.uniform(-0.05, 0.30, n_assets)           # -5 % … +30 % p.a.
    loadings = rng.normal(0.0, 0.15, (n_assets, 2))              # Faktorladungen B
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.05               # alle hängen am Markt
    idio_vol = rng.uniform(0.10, 0.35, n_assets)                 # Eigenvolatilität p.a.

    true_cov = loadings @ loadings.T + np.diag(idio_vol**2)      # jährliche Kovarianz
    daily_cov = true_cov / trading_days
    daily_drift = annual_drift / trading_days - 0.5 * np.diag(daily_cov)  # Itô-Korrektur

    # Korrelierte Log-Renditen über Cholesky-Zerlegung: r = drift + L · z
    chol = np.linalg.cholesky(daily_cov)
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T
    log_returns = daily_drift + shocks

    start_prices = rng.uniform(20, 300, n_assets)
    prices = start_prices * np.exp(np.vstack([np.zeros(n_assets), np.cumsum(log_returns, axis=0)]))

    # Schätzung wie bei echten Daten: aus den beobachteten Tagesrenditen
    simple_returns = prices[1:] / prices[:-1] - 1
    mu = simple_returns.mean(axis=0) * trading_days
    sigma = np.cov(simple_returns, rowvar=False) * trading_days

    return MarketData(tickers, prices, mu, sigma)


# =============================================================================
# 2. QUBO aufbauen
# =============================================================================

def build_portfolio_qubo(mu: np.ndarray, sigma: np.ndarray, k: int, q: float,
                         penalty: float | None = None) -> tuple[np.ndarray, float, float]:
    """
    Baut die symmetrische QUBO-Matrix Q und eine Konstante c, sodass

        E(x) = xᵀ Q x + c  =  q·xᵀΣx − μᵀx + P·(Σx_i − K)²

    Herleitung Term für Term (immer mit x_i² = x_i):

    (a) Risiko  q·xᵀΣx
        Ist bereits quadratisch  →  Q += q·Σ
        (Die Diagonale Σ_ii wird automatisch zum linearen Term q·Σ_ii·x_i.)

    (b) Rendite  −μᵀx
        Linear  →  auf die Diagonale:  Q_ii += −μ_i

    (c) Strafe  P·(Σ_i x_i − K)²
        Ausmultiplizieren:
            (Σ_i x_i)²  = Σ_i x_i²  + Σ_{i≠j} x_i x_j  =  Σ_i x_i + Σ_{i≠j} x_i x_j
            −2K·Σ_i x_i
            +K²
        Also:
            Q_ii += P·(1 − 2K)
            Q_ij += P        für alle i ≠ j
            c    += P·K²

    Wahl von P: Ändert man die Auswahl um genau eine Aktie i, verändert sich
    Rendite+Risiko um höchstens |μ_i| + q·(Σ_ii + 2·Σ_j|Σ_ij|). P muss größer
    sein als dieser maximal mögliche "Gewinn", sonst würde der Optimierer
    lieber die Nebenbedingung brechen. Zu groß sollte P aber auch nicht sein:
    dann dominiert die Strafe die Energielandschaft und Heuristiken (v. a. QAOA)
    finden das eigentliche Optimum schlechter. Faktor 1.5 ist ein guter Kompromiss.
    """
    n = len(mu)
    if penalty is None:
        max_gain = np.abs(mu) + q * (np.diag(sigma) + 2 * np.abs(sigma).sum(axis=1))
        penalty = 1.5 * max_gain.max()

    Q = q * sigma.copy()                        # (a)
    Q[np.diag_indices(n)] -= mu                 # (b)
    Q += penalty * (np.ones((n, n)) - np.eye(n))  # (c) Off-Diagonale
    Q[np.diag_indices(n)] += penalty * (1 - 2 * k)  # (c) Diagonale
    constant = penalty * k**2                   # (c) Konstante

    return Q, constant, penalty


def qubo_energy(x: np.ndarray, Q: np.ndarray, constant: float = 0.0) -> float:
    return float(x @ Q @ x + constant)


def qubo_to_ising(Q: np.ndarray, constant: float) -> tuple[SparsePauliOp, float]:
    """
    Übersetzt das QUBO in einen Ising-Hamiltonoperator für Qubits.

    Ein Qubit liefert beim Messen |0⟩ oder |1⟩; der Pauli-Z-Operator hat darauf
    die Eigenwerte z = +1 bzw. z = −1. Die Zuordnung zum Bit x lautet daher

        x_i = (1 − z_i) / 2          (|0⟩ → x=0,  |1⟩ → x=1)

    Einsetzen in E(x) = Σ_i Q_ii x_i + Σ_{i≠j} Q_ij x_i x_j + c ergibt

        H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + offset

    mit
        h_i    = −½ · Σ_j Q_ij                  (Zeilensumme)
        J_ij   =  ½ · Q_ij                      (für i < j, Q symmetrisch)
        offset =  ¼ · (Spur(Q) + Σ_ij Q_ij) + c

    Der Grundzustand (kleinster Eigenwert) von H entspricht genau der
    optimalen QUBO-Lösung. Aufgabe von QAOA: diesen Grundzustand finden.
    """
    n = Q.shape[0]
    h = -0.5 * Q.sum(axis=1)
    offset = 0.25 * (np.trace(Q) + Q.sum()) + constant

    terms = []
    for i in range(n):
        terms.append(("Z", [i], h[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                terms.append(("ZZ", [i, j], 0.5 * Q[i, j]))

    hamiltonian = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    return hamiltonian, float(offset)


# =============================================================================
# Auswertung / Ausgabe
# =============================================================================

def describe_portfolio(x: np.ndarray, data: MarketData) -> str:
    chosen = np.flatnonzero(x)
    if len(chosen) == 0:
        return "(leer)"
    w = x / x.sum()                                    # gleich gewichtet
    ret = w @ data.mu
    vol = np.sqrt(w @ data.sigma @ w)
    names = ", ".join(data.tickers[i] for i in chosen)
    return f"{names:<34} Rendite {ret:+7.2%}  Volatilität {vol:6.2%}  Sharpe {ret / vol:5.2f}"


def print_distribution(probabilities: dict[tuple, float], x_opt: np.ndarray | None, k: int,
                       Q: np.ndarray, constant: float,
                       p_opt: float | None = None, p_feasible: float | None = None) -> None:
    """
    Kennzahlen einer Messverteilung (simuliert oder von der Hardware).

    probabilities: Bitstring → Wahrscheinlichkeit, absteigend sortiert.
    x_opt:         bekannte optimale Lösung (None, wenn keine Referenz vorliegt).
    p_opt, p_feasible: optional exakt vorberechnet, falls probabilities nur die
                   wahrscheinlichsten Zustände enthält.
    """
    n = len(next(iter(probabilities)))
    if p_feasible is None:
        p_feasible = sum(p for x, p in probabilities.items() if sum(x) == k)
    if x_opt is not None:
        if p_opt is None:
            p_opt = probabilities.get(tuple(x_opt), 0.0)
        print(f"  P(optimales Depot messen) = {p_opt:.2%}   (Zufall: {1 / 2**n:.2%})")
    print(f"  P(gültiges Depot, K = {k}) = {p_feasible:.1%}   (Zufall: {math.comb(n, k) / 2**n:.1%})")
    print("  Top-5 Messergebnisse:")
    for x, p in list(probabilities.items())[:5]:
        x = np.array(x)
        marker = "  ← optimal" if x_opt is not None and np.array_equal(x, x_opt) else ""
        print(f"    {''.join(map(str, x))}  p = {p:6.2%}  E = {qubo_energy(x, Q, constant):+.4f}{marker}")


def plot_prices(data: MarketData, x_opt: np.ndarray, title_suffix: str = "") -> None:
    """Normierte Kursverläufe; Aktien im Depot durchgezogen, der Rest gepunktet."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(data.tickers):
        style = "-" if x_opt[i] else ":"
        ax.plot(data.prices[:, i] / data.prices[0, i], style, label=t)
    ax.set_title(f"Normierte Kursverläufe (durchgezogen = im Depot{title_suffix})")
    ax.set_xlabel("Handelstag")
    ax.set_ylabel("Kurs / Startkurs")
    ax.legend(ncol=4)
    plt.tight_layout()


# =============================================================================
# Gemeinsame Kommandozeile und Problemaufbau
# =============================================================================

def make_parser(description: str) -> argparse.ArgumentParser:
    """Argumente, die jedes Skript kennt. Skripte können eigene ergänzen."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-assets", type=int, default=8, help="Anzahl Aktien N (= Qubits bei QAOA)")
    parser.add_argument("--k", type=int, default=None, help="Anzahl auszuwählender Aktien K (Standard: N/2)")
    parser.add_argument("--q", type=float, default=1.0, help="Risikoaversion")
    parser.add_argument("--days", type=int, default=504, help="Handelstage Historie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Kursverläufe plotten")
    return parser


@dataclass
class Problem:
    data: MarketData
    Q: np.ndarray
    constant: float
    penalty: float
    k: int
    rng: np.random.Generator

    @property
    def n(self) -> int:
        return len(self.data.tickers)


def setup_problem(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Problem:
    """
    Erzeugt Marktdaten und QUBO aus den Kommandozeilenargumenten und gibt eine
    Übersicht aus. Die Marktdaten werden als Erstes aus dem Zufallsgenerator
    gezogen – deshalb sind sie bei gleichem --seed in allen Skripten identisch.
    """
    n = args.n_assets
    k = args.k if args.k is not None else n // 2
    if n < 2:
        parser.error("--n-assets muss mindestens 2 sein")
    if not 1 <= k <= n:
        parser.error(f"--k muss zwischen 1 und N = {n} liegen")

    rng = np.random.default_rng(args.seed)
    data = generate_market_data(n, args.days, rng)

    print("=" * 78)
    print(f"Zufällige Marktdaten: {n} Aktien, {args.days} Handelstage, Seed {args.seed}")
    print("=" * 78)
    vols = np.sqrt(np.diag(data.sigma))
    for t, p0, p1, m, v in zip(data.tickers, data.prices[0], data.prices[-1], data.mu, vols):
        print(f"  {t:<6}  {p0:8.2f} → {p1:8.2f}   μ = {m:+7.2%}   σ = {v:6.2%}")

    Q, constant, penalty = build_portfolio_qubo(data.mu, data.sigma, k, args.q)
    print(f"\nQUBO: {n}×{n}-Matrix, wähle K = {k}, "
          f"Risikoaversion q = {args.q}, Strafe P = {penalty:.3f}")
    print(f"Suchraum: 2^{n} = {2**n:,} Bitstrings, davon {math.comb(n, k):,} gültig")

    return Problem(data, Q, constant, penalty, k, rng)
