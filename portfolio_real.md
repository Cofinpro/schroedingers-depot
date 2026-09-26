# Real-world portfolio optimization, step by step

Run it with `uv run python portfolio_real.py`. The first run downloads prices into
`data/prices.csv`. Later runs reuse that file and work offline. Set `REFRESH_DATA = True`
to download fresh prices. The whole run takes about 20 seconds.

**The question:** you have money to invest in 8 assets. Which ones should you buy,
and how much of each?

| Ticker | What it is |
|---|---|
| AAPL, MSFT | tech stocks |
| JPM | a bank |
| JNJ | healthcare |
| XOM | oil |
| PG | consumer goods (soap, diapers) |
| GLD | an ETF that tracks gold |
| TLT | an ETF of long-term US government bonds |

This is the same idea as `portfolio.py` (Kata 8), with four upgrades:
**real prices**, **honest statistics**, **a split into "pick" (quantum) and "how much"
(classical)**, and **a backtest** that checks whether the method would actually have worked.

---

## Step 1 · Real prices

`download_prices()` asks Yahoo Finance for 5 years of daily **adjusted close** prices.
"Adjusted" means dividends and stock splits are already folded in. For example, if Apple
splits 1 share into 4, the old prices are divided by 4, so the chart doesn't show a fake
75% crash.

Only the days where *all 8* assets have a price are kept, which gives about 1255 trading days.
A year has about 252 trading days, because weekends and holidays have no trading.

## Step 2 · Returns

We don't care about the price itself ($200 vs $50). We care about how much it *moves*:

$$r_t = \frac{\text{price}_t}{\text{price}_{t-1}} - 1$$

If Apple goes from 200 to 202, then r = 0.01 = +1%. Each day gives 8 numbers, one per asset.

## Step 3 · Estimates: expected return μ and risk Σ

To look into the future, we only have the past. We use the last 252 days.

**Expected return μ ("mu").** This is the average daily return × 252, giving a yearly number.
If AAPL averaged +0.1% per day, that's about 25% per year.

**Risk = covariance matrix Σ ("Sigma").** This is an 8×8 table:
- The **diagonal** Σᵢᵢ holds each asset's *variance*, meaning how much it wobbles.
  Its square root is the **volatility**. 20% volatility means a typical year lands
  roughly ±20% around the expected return.
- The **off-diagonal** Σᵢⱼ says whether two assets wobble *together*.
  Dividing by both volatilities gives the **correlation**, from −1 to +1:
  - +1: they always move together, so holding both doesn't reduce risk.
  - 0: they're unrelated, so holding both halves some of the wobble.
  - −1: when one falls, the other rises, which is perfect protection.

The risk of a whole portfolio with weights w (for example w = [0.25, 0, 0.4, …], summing to 1) is

$$\text{variance} = w^T \Sigma w = \sum_i \sum_j w_i\, w_j\, \Sigma_{ij}$$

That's every pair of assets, weighted by how much you hold of each and how they move together.
This formula is *why diversification works*: pairs with low or negative correlation
make the total smaller.

**Shrinkage: the "real-world" part.** One year of data is noisy. An asset that went up 60%
last year was partly lucky, and a naive optimizer would put all your money in it.
Professionals therefore pull the estimates toward something boring:
- **μ** is moved halfway toward the average of all 8 assets.
- **Σ** uses the **Ledoit–Wolf** method. It blends the measured table with a "target" table
  in which every asset is equally risky and nothing is correlated. The blend amount is
  calculated from how noisy the data is: the more noise, the more shrinkage.
  The script prints this amount (about 20%).

## Step 4 · QUANTUM: pick the best 4 of 8 assets (QAOA)

**The problem as bits.** Let xᵢ = 1 if we buy asset i, and 0 if not. With 8 assets there are
2⁸ = 256 possible choices, and exactly 70 of them pick 4 assets.

Suppose we put equal money in each picked asset (w = x / K, with K = 4). The cost of a choice is then:

$$\text{cost}(x) = \underbrace{\lambda\, w^T\Sigma w}_{\text{risk, weighted by risk aversion}} \;-\; \underbrace{\mu^T w}_{\text{return}} \;+\; \underbrace{P\,(\textstyle\sum_i x_i - K)^2}_{\text{penalty: not exactly 4 picked}}$$

Every term is at most "x times x", so the cost fits the form **xᵀAx + bᵀx**.
That form is called a **QUBO** (Quadratic Unconstrained Binary Optimization).
It's the standard input format for quantum optimizers. The penalty P is set large enough
that breaking the "exactly 4" rule is never worth it.

**From bits to qubits (Ising form).** Measuring a qubit with Z gives a spin z = +1 for |0⟩
and −1 for |1⟩, so we substitute x = (1 − z)/2. The cost becomes

$$H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{constant}$$

`to_ising()` does this algebra. It's the same kind of diagonal cost layer you used in
Kata 8, but now built from real gates: **rz** for each hᵢ term and **rzz** for each
Jᵢⱼ pair. That makes 8 + 28 = 36 gates per layer, which is what real IBM hardware
would run. The script checks that this conversion matches the brute-force costs exactly.

**The QAOA circuit** (Quantum Approximate Optimization Algorithm) has three parts:
1. `h` on all 8 qubits gives an equal blend of all 256 choices (Kata 1).
2. The **cost layer** turns each choice's weight by an angle proportional to its cost × γ ("gamma").
   Like the hidden minus sign in Grover (Kata 5), this changes nothing you can measure yet.
3. The **mixer layer** applies `rx(2β)` to every qubit. It lets weight flow between
   neighbouring choices, so the hidden angles turn into *interference*: good choices
   grow and bad ones shrink.

Steps 2 and 3 repeat p = 3 times. That gives 6 angles (γ₁, β₁, γ₂, β₂, γ₃, β₃).

**Tuning the angles (hybrid loop).** The classical optimizer COBYLA proposes angles, the
quantum circuit is run, the result is scored, and the loop repeats about 200 times.
It restarts 3 times from random angles and keeps the best result.

The score is **CVaR** (Conditional Value at Risk). It is the average cost of only the
*best 25%* of outcomes. Why not the plain average? We only need the circuit to
*sometimes* output the best answer, because we can measure many times and keep the cheapest.
CVaR rewards exactly that, and it works much better than the average in practice.

**Measure and keep the best.** We sample the tuned circuit 1024 times (1024 "shots")
and keep the cheapest choice we saw.

**Honesty check.** With only 256 choices, a normal computer can simply try them all
("brute force"). The script does that too and prints whether QAOA agrees. In the latest
run it did in **16 of 16 quarters**. The tuned circuit outputs the best choice directly
about 3.6% of the time, versus 0.4% for random guessing. Quantum computers only become
interesting at hundreds of assets, where brute force is impossible (2¹⁰⁰ ≈ 10³⁰ choices).
This script is the full real workflow at a size where we can still check the answer.

## Step 5 · CLASSICAL: how much money in each picked asset

Now that the 4 assets are chosen, the weights (the percentage in each) are continuous
numbers, which classical optimizers handle very well. We maximize the **Sharpe ratio**:

$$\text{Sharpe} = \frac{\text{expected return} - \text{risk-free rate}}{\text{volatility}} = \frac{\mu^T w - 4\%}{\sqrt{w^T\Sigma w}}$$

This is extra return (above safe cash at 4%) per unit of wobble. Rules:
weights ≥ 0 (no short selling), they sum to 100%, and no asset gets more than 40%.
The cap is a real-world safety rule: never bet too much on one estimate.

This split of *discrete choice → quantum* and *continuous amounts → classical* is how
current quantum-finance research actually combines the two.

## Step 6 · Walk-forward backtest: would it have worked?

A strategy that looks great on the data it was built from proves nothing (this is called
overfitting). So we replay history honestly:
1. At each quarter (every 63 days), look **only at the past year** and choose weights
   with steps 3–5.
2. Hold those weights for the quarter. Weights drift as prices move: if AAPL rises,
   it becomes a bigger slice.
3. Pay **0.1% trading costs** on everything bought or sold (the "turnover").
4. Repeat. That's 16 quarters, about 4 years.

We compare against three simple strategies:

| Strategy | Idea |
|---|---|
| Equal weight (1/N) | 12.5% in each, no thinking at all |
| Min variance | lowest possible wobble, ignores returns |
| Max Sharpe (all 8) | step 5 without the quantum pick |
| QAOA pick + Max Sharpe | the full pipeline |

The metrics are:
- **yearly**: average growth per year.
- **volatility**: how bumpy the ride was.
- **Sharpe**: return per unit of bumpiness, where higher is better.
- **max drop**: the worst fall from a peak.
- **turnover**: how much gets traded each quarter, which drives costs.

**Latest run:**

| Strategy | Yearly | Volatility | Sharpe | Max drop |
|---|---|---|---|---|
| Equal weight | 20.7% | 10.7% | **1.56** | 10.5% |
| Min variance | 15.0% | 9.5% | 1.16 | **8.7%** |
| Max Sharpe (all 8) | 18.0% | 12.9% | 1.09 | 13.0% |
| QAOA pick + Max Sharpe | **21.5%** | 13.8% | 1.27 | 14.7% |

**How to read this honestly:** the full pipeline earned the most, but it took more risk
to get there. Simple equal weighting had the best return per unit of risk. That's a
famous real-world result: estimated μ is so noisy that "1/N" is very hard to beat.
The quantum step didn't make the pick *better* than brute force. It found the *same*
answer through a different machine. The value of this code is the workflow, not
a money machine.

## Step 7 · Today's recommendation

Steps 3–5 are run on the most recent year, and the plots show:
- **left:** how $1 grew under each strategy during the backtest.
- **right:** every asset's risk vs return, the **efficient frontier** (the lowest risk
  you can get for each level of return), and the recommended portfolio as a red star.

---

## Things to try

- `PICK_K = 3` or `5`: concentrate or diversify more.
- `RISK_AVERSION = 10`: the quantum pick favours calmer assets (watch TLT and PG).
- `QAOA_LAYERS = 1`: does QAOA still find the best pick? What happens to the 3.6%?
- Add a ticker to `TICKERS` (9 qubits = 512 choices) and set `REFRESH_DATA = True`.
- `MAX_WEIGHT = 1.0`: remove the safety cap and watch the optimizer go all-in.

## Where real systems go further

- **Real hardware:** run the circuit through `qiskit-ibm-runtime` (Sampler) on an IBM
  device, or add noise with FakeBrisbane as in Kata 7.
- **Better return estimates:** factor models or Black–Litterman, instead of simple shrinkage.
- **More constraints:** sector limits, minimum trade sizes, and taxes.

*Educational code, not investment advice. Prices come from Yahoo Finance's public chart
endpoint for personal study.*
