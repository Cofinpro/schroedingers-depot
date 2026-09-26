"""
Real-world portfolio optimization: classical finance + a quantum (QAOA) step.

Pipeline (each step is a section below, and explained in portfolio_real.md):
  1. Download 5 years of real daily prices (Yahoo Finance), cache them in data/
  2. Turn prices into daily returns
  3. Estimate expected return and risk from the past year (with "shrinkage")
  4. QUANTUM: pick the best K of the N assets with QAOA (a QUBO problem)
  5. CLASSICAL: split the money between the picked assets (max Sharpe ratio)
  6. Backtest: repeat 3-5 every quarter, only ever using past data,
     and compare against simple strategies
  7. Today's recommended portfolio

Educational code, not investment advice.
"""
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# --- Settings ---------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "GLD", "TLT"]
# tech, tech, bank, healthcare, oil, consumer goods, gold ETF, long US bonds ETF

PICK_K = 4             # the quantum step picks exactly this many assets
LOOKBACK = 252         # days of history used for each decision (252 trading days = 1 year)
REBALANCE_EVERY = 63   # re-decide every quarter (63 trading days)
MAX_WEIGHT = 0.40      # never put more than 40% in one asset
RISK_FREE = 0.04       # yearly return of "cash" (treasury bills), for the Sharpe ratio
RISK_AVERSION = 2.0    # lambda in the QUBO: how much the picking step hates risk
COST_PER_TRADE = 0.001 # 0.1% trading cost on every dollar bought or sold
QAOA_LAYERS = 3        # p: how many (cost layer + mixer layer) rounds
CVAR_ALPHA = 0.25      # QAOA scores the best 25% of outcomes (CVaR), see step 4
SHOTS = 1024           # how many times we "measure" the final quantum state
DATA_FILE = Path("data/prices.csv")
REFRESH_DATA = False   # True: download fresh prices even if data/prices.csv exists
SHOW_PLOTS = True

N = len(TICKERS)
DAYS_PER_YEAR = 252
rng = np.random.default_rng(seed=7)


# --- 1. Real prices ---------------------------------------------------
def download_prices():
    """Daily 'adjusted close' prices (dividends and splits folded in) for 5 years."""
    columns = {}
    for ticker in TICKERS:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5y&interval=1d"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        chart = json.load(urllib.request.urlopen(request, timeout=30))["chart"]["result"][0]
        days = [datetime.fromtimestamp(t, timezone.utc).date().isoformat() for t in chart["timestamp"]]
        closes = chart["indicators"]["adjclose"][0]["adjclose"]
        columns[ticker] = {d: c for d, c in zip(days, closes) if c is not None}
        time.sleep(0.5)  # be polite to the server
    # keep only days where every asset has a price
    dates = sorted(set.intersection(*(set(col) for col in columns.values())))
    DATA_FILE.parent.mkdir(exist_ok=True)
    with DATA_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + TICKERS)
        for d in dates:
            writer.writerow([d] + [columns[t][d] for t in TICKERS])


def load_prices():
    if REFRESH_DATA or not DATA_FILE.exists():
        print("Downloading prices from Yahoo Finance ...")
        download_prices()
    with DATA_FILE.open() as f:
        rows = list(csv.reader(f))
    assert rows[0][1:] == TICKERS, "data/prices.csv has other tickers: set REFRESH_DATA = True"
    dates = [r[0] for r in rows[1:]]
    prices = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return dates, prices


# --- 2. Returns -------------------------------------------------------
# r[t] = price[t] / price[t-1] - 1   ("AAPL went up 1.2% today" -> 0.012)
def daily_returns(prices):
    return prices[1:] / prices[:-1] - 1


# --- 3. Estimates: expected return (mu) and risk (Sigma) -------------
def estimate(returns):
    """Yearly expected returns mu and covariance Sigma from a window of daily returns.

    Past data is noisy, so both estimates are 'shrunk' toward something boring:
      mu    -> halfway toward the average of all assets (past winners are often luck)
      Sigma -> toward 'every asset equally risky, no correlation' (Ledoit-Wolf)
    """
    T = len(returns)
    mu_sample = returns.mean(axis=0)
    mu = 0.5 * mu_sample + 0.5 * mu_sample.mean()

    X = returns - mu_sample                      # centered returns, T x N
    S = X.T @ X / T                              # sample covariance
    target = np.trace(S) / N * np.eye(N)         # the boring target
    d2 = np.sum((S - target) ** 2)               # how far S is from the target
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / T ** 2  # how noisy S is
    shrink = min(b2, d2) / d2                    # noisier -> shrink more (0..1)
    Sigma = shrink * target + (1 - shrink) * S

    return mu * DAYS_PER_YEAR, Sigma * DAYS_PER_YEAR, shrink


# --- 4. QUANTUM: pick K assets with QAOA ------------------------------
def bits_of(i):
    """Basis state number i -> 0/1 per asset (qubit k = asset k, Qiskit order)."""
    return np.array([(i >> k) & 1 for k in range(N)])


ALL_BITS = np.array([bits_of(i) for i in range(2 ** N)])  # all 256 possible picks


def qubo(mu, Sigma):
    """The picking problem as a QUBO: cost(x) = x^T A x + b^T x, with x_i in {0, 1}.

    With x = which assets we hold (equal money in each, w = x / K):
      RISK_AVERSION * risk(w)  -  return(w)  +  penalty * (number picked - K)^2
    """
    penalty = 2 * (np.abs(mu).max() + RISK_AVERSION * np.abs(Sigma).max())
    A = RISK_AVERSION * Sigma / PICK_K ** 2 + penalty * np.ones((N, N))
    b = -mu / PICK_K - 2 * penalty * PICK_K
    return A, b


def qubo_costs(A, b):
    """Cost of every one of the 256 picks (the classical brute-force answer key)."""
    return np.einsum("si,ij,sj->s", ALL_BITS, A, ALL_BITS) + ALL_BITS @ b


def to_ising(A, b):
    """Rewrite x_i in {0,1} as spins z_i in {+1,-1} (x = (1 - z)/2), which is what
    qubits measure with Z. Gives cost = sum h_i z_i + sum_{i<j} J_ij z_i z_j + const."""
    J = A / 2
    h = -b / 2 - A.sum(axis=1) / 2
    return h, np.triu(J, k=1)


def qaoa_template(h, J):
    """p layers of: cost layer exp(-i gamma H) (rz/rzz gates), then mixer exp(-i beta X)."""
    scale = max(np.abs(h).max(), np.abs(J).max())  # keep angles in a friendly range
    gammas, betas = ParameterVector("gamma", QAOA_LAYERS), ParameterVector("beta", QAOA_LAYERS)
    qc = QuantumCircuit(N)
    qc.h(range(N))                                # equal blend of all 256 picks
    for gamma, beta in zip(gammas, betas):
        for i in range(N):
            qc.rz(2 * gamma * h[i] / scale, i)    # one-asset terms
        for i in range(N):
            for j in range(i + 1, N):
                qc.rzz(2 * gamma * J[i, j] / scale, i, j)  # pair terms (risk + budget)
        qc.rx(2 * beta, range(N))                 # mixer: let weight flow between picks
    return qc


def cvar(probs, costs):
    """Average cost of the best CVAR_ALPHA share of outcomes (lower is better)."""
    order = np.argsort(costs)
    p, c = probs[order], costs[order]
    kept = np.minimum(p, np.maximum(0, CVAR_ALPHA - (np.cumsum(p) - p)))
    return kept @ c / CVAR_ALPHA


def quantum_pick(mu, Sigma):
    A, b = qubo(mu, Sigma)
    costs = qubo_costs(A, b)
    h, J = to_ising(A, b)
    template = qaoa_template(h, J)

    def score(angles):
        probs = Statevector(template.assign_parameters(angles)).probabilities()
        return cvar(probs, costs)

    # the classical optimizer tunes the angles; a few random starts avoid bad valleys
    best = min((minimize(score, rng.uniform(0, np.pi, 2 * QAOA_LAYERS), method="COBYLA",
                         options={"maxiter": 200}) for _ in range(3)), key=lambda r: r.fun)

    # "measure" the tuned circuit SHOTS times and keep the cheapest pick we saw
    state = Statevector(template.assign_parameters(best.x))
    state.seed(int(rng.integers(1 << 31)))
    seen = [int(s, 2) for s in state.sample_counts(SHOTS)]
    pick = min(seen, key=lambda i: costs[i])
    return {
        "bits": bits_of(pick),
        "found_optimum": pick == int(np.argmin(costs)),
        "optimum_prob": state.probabilities()[np.argmin(costs)],
        "circuit": template.assign_parameters(best.x),
    }


# --- 5. CLASSICAL: how much money in each picked asset ----------------
def max_sharpe(mu, Sigma, allowed):
    """Weights (sum 1, each 0..MAX_WEIGHT, only in allowed assets) with the best Sharpe
    ratio = (return - risk-free) / volatility: return earned per unit of wobble."""
    idx = np.flatnonzero(allowed)
    cap = max(MAX_WEIGHT, 1 / len(idx))  # the cap must still let weights add up to 1

    def neg_sharpe(w):
        return -(w @ mu[idx] - RISK_FREE) / np.sqrt(w @ Sigma[np.ix_(idx, idx)] @ w)

    result = minimize(neg_sharpe, np.full(len(idx), 1 / len(idx)), method="SLSQP",
                      bounds=[(0, cap)] * len(idx),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    weights = np.zeros(N)
    weights[idx] = result.x
    return weights


def min_variance(Sigma):
    """Weights with the lowest possible volatility (ignores returns completely)."""
    result = minimize(lambda w: w @ Sigma @ w, np.full(N, 1 / N), method="SLSQP",
                      bounds=[(0, MAX_WEIGHT)] * N,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return result.x


STRATEGIES = {
    "Equal weight (1/N)": lambda mu, Sigma: np.full(N, 1 / N),
    "Min variance": lambda mu, Sigma: min_variance(Sigma),
    "Max Sharpe (all 8)": lambda mu, Sigma: max_sharpe(mu, Sigma, np.ones(N)),
    "QAOA pick + Max Sharpe": lambda mu, Sigma: qaoa_then_sharpe(mu, Sigma),
}
qaoa_log = []  # did QAOA find the true best pick? one entry per quarter


def qaoa_then_sharpe(mu, Sigma):
    pick = quantum_pick(mu, Sigma)
    qaoa_log.append(pick["found_optimum"])
    return max_sharpe(mu, Sigma, pick["bits"])


# --- 6. Walk-forward backtest -----------------------------------------
def backtest(returns, choose_weights):
    """Every quarter: look at the past year only, choose weights, then hold them
    (letting them drift with prices) until the next quarter. Pay trading costs."""
    value, values, turnovers = 1.0, [], []
    held = np.zeros(N)
    for start in range(LOOKBACK, len(returns), REBALANCE_EVERY):
        mu, Sigma, _ = estimate(returns[start - LOOKBACK:start])  # past data only!
        target = choose_weights(mu, Sigma)
        turnover = np.abs(target - held).sum()
        value *= 1 - COST_PER_TRADE * turnover
        turnovers.append(turnover)
        held = target
        for r in returns[start:start + REBALANCE_EVERY]:
            growth = held * (1 + r)             # each asset's slice grows with its return
            value *= growth.sum()
            held = growth / growth.sum()        # weights drift
            values.append(value)
    return np.array(values), np.mean(turnovers[1:])


def report(values, turnover):
    daily = np.diff(np.concatenate([[1.0], values])) / np.concatenate([[1.0], values[:-1]])
    years = len(values) / DAYS_PER_YEAR
    cagr = values[-1] ** (1 / years) - 1
    vol = daily.std() * np.sqrt(DAYS_PER_YEAR)
    sharpe = (cagr - RISK_FREE) / vol
    max_drawdown = (1 - values / np.maximum.accumulate(values)).max()
    return cagr, vol, sharpe, max_drawdown, turnover


# --- Run everything ---------------------------------------------------
if __name__ == "__main__":
    dates, prices = load_prices()
    returns = daily_returns(prices)
    print(f"Step 1-2: {len(prices)} trading days of prices, {dates[0]} to {dates[-1]}, "
          f"{N} assets: {', '.join(TICKERS)}\n")

    # Steps 3-5 on the latest year, shown in detail
    mu, Sigma, shrink = estimate(returns[-LOOKBACK:])
    vol = np.sqrt(np.diag(Sigma))
    print(f"Step 3: estimates from the last {LOOKBACK} days (covariance shrinkage {shrink:.0%})")
    print("  asset   exp. return   volatility")
    for t, m, v in zip(TICKERS, mu, vol):
        print(f"  {t:6} {m:10.1%} {v:12.1%}")
    corr = Sigma / np.outer(vol, vol)
    pairs = [(corr[i, j], TICKERS[i], TICKERS[j]) for i in range(N) for j in range(i + 1, N)]
    print("  most related pair: {1}+{2} (correlation {0:.2f}), most independent: {4}+{5} ({3:.2f})\n"
          .format(*max(pairs), *min(pairs)))

    A, b = qubo(mu, Sigma)
    costs = qubo_costs(A, b)
    pick = quantum_pick(mu, Sigma)
    brute = bits_of(int(np.argmin(costs)))
    print(f"Step 4: QAOA with {N} qubits, {QAOA_LAYERS} layers, "
          f"{pick['circuit'].size()} gates, {2 ** N} possible picks")
    print("  QAOA picked:       ", ", ".join(t for t, x in zip(TICKERS, pick["bits"]) if x))
    print("  brute force best:  ", ", ".join(t for t, x in zip(TICKERS, brute) if x))
    print(f"  chance the circuit outputs the best pick directly: {pick['optimum_prob']:.1%} "
          f"(random guessing: {1 / 2 ** N:.1%})\n")

    weights = max_sharpe(mu, Sigma, pick["bits"])
    print(f"Step 5 + 7: today's recommendation, money split over the picked assets "
          f"(max Sharpe, cap {MAX_WEIGHT:.0%})")
    for t, w in zip(TICKERS, weights):
        if w > 0.005:
            print(f"  {t:6} {w:6.1%}  {'#' * round(w * 50)}")
    print(f"  expected return {weights @ mu:.1%}, volatility {np.sqrt(weights @ Sigma @ weights):.1%}\n")

    print(f"Step 6: walk-forward backtest, rebalancing every {REBALANCE_EVERY} days "
          f"(QAOA runs {len(range(LOOKBACK, len(returns), REBALANCE_EVERY))} times, takes a minute) ...")
    curves = {}
    print(f"  {'strategy':25} {'yearly':>7} {'volat.':>7} {'Sharpe':>7} {'max drop':>9} {'turnover':>9}")
    for name, strategy in STRATEGIES.items():
        values, turnover = backtest(returns, strategy)
        curves[name] = values
        cagr, v, sharpe, dd, to = report(values, turnover)
        print(f"  {name:25} {cagr:7.1%} {v:7.1%} {sharpe:7.2f} {dd:9.1%} {to:9.0%}")
    print(f"  QAOA found the brute-force best pick in {sum(qaoa_log)} of {len(qaoa_log)} quarters")

    if SHOW_PLOTS:
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
        for name, values in curves.items():
            left.plot(values, label=name)
        left.set_title("Backtest: $1 grows to ...")
        left.set_xlabel(f"trading days after {dates[LOOKBACK + 1]}")
        left.legend()

        # efficient frontier: lowest risk for each target return (all assets allowed)
        frontier = []
        for target in np.linspace(mu.min(), mu.max(), 30):
            r = minimize(lambda w: w @ Sigma @ w, np.full(N, 1 / N), method="SLSQP",
                         bounds=[(0, 1)] * N,
                         constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                      {"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
            if r.success:
                frontier.append((np.sqrt(r.fun), target))
        right.plot(*zip(*frontier), "k-", label="efficient frontier")
        right.scatter(vol, mu)
        for t, x, y in zip(TICKERS, vol, mu):
            right.annotate(t, (x, y))
        right.scatter(np.sqrt(weights @ Sigma @ weights), weights @ mu, marker="*", s=200,
                      color="red", label="recommended")
        right.set_xlabel("volatility (risk)")
        right.set_ylabel("expected yearly return")
        right.set_title("Risk vs return, last year")
        right.legend()
        plt.tight_layout()
        plt.show()
