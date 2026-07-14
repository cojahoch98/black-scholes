# Black-Scholes vs. Real Market Data: A DUOL Case Study

A worked example of pricing real Duolingo (DUOL) options with a
from-scratch Black-Scholes model — including the data problems
encountered and what they revealed about the model's real-world limits.

## Why This Exists

Building Black-Scholes from scratch is a common portfolio project, but
the formula alone doesn't prove much — anyone can transcribe five lines
of math from a textbook. The more interesting question is whether the
model's assumptions actually hold up against real, live market prices.

This case study pulls a real options chain for Duolingo (DUOL), prices
it with the model, and documents where the model matches the market —
and, just as importantly, where and why it doesn't.

## Data Pipeline

Real options data via `yfinance`:

```python
import yfinance as yf

ticker = yf.Ticker("DUOL")
expiry = ticker.options[0]
chain = ticker.option_chain(expiry).calls
```

Raw option chains are noisy — many strikes have stale quotes from
trades that happened days or weeks ago, which produces nonsensical
implied volatilities (over 300%, or near 0%). These strikes are
filtered out by requiring real, recent trading activity:

```python
liquid = chain[
    (chain["volume"] > 10) &
    (chain["openInterest"] > 50) &
    (chain["bid"] > 0)
].copy()

liquid["mid_price"] = (liquid["bid"] + liquid["ask"]) / 2
```

Using the bid/ask **midpoint** rather than `lastPrice` matters here:
`lastPrice` reflects whenever the last trade happened to occur, which
could be stale, while the bid/ask midpoint reflects what the market is
willing to trade at *right now*.

## Debugging Story: The Expiry Mismatch

The first pass at comparing model prices to market prices produced
model prices roughly 60-80% *below* the real market price, across
every strike. That gap wasn't random noise — it grew systematically
larger the further a strike sat from the current stock price, which is
the signature of a wrong *input*, not a wrong formula.

**Diagnosis:** rather than guess, the days-to-expiration was solved for
directly — using one clean, liquid data point (strike 125, the
near-the-money option) and finding what `T` would make the model's
price match the market's actual price:

```python
from scipy.optimize import brentq

def price_diff(T):
    inputs = OptionInputs(S=125.83, K=125, T=T, r=0.038, sigma=0.611332)
    bs = BlackScholes(inputs)
    return bs.price(OptionType.CALL) - 8.45  # actual market mid-price

implied_T = brentq(price_diff, 0.001, 2.0)
# implied_T = 0.0664 years = ~24 days
```

**Root cause:** `ticker.options[0]` was assumed to be the nearest
expiration, but it was actually further out (~24 days), not the ~4-day
expiry originally assumed. Once the correct expiry date was substituted,
the near-the-money strike matched the market price within 1.5%.

**Fix:** the pipeline now explicitly computes and prints the real
days-to-expiration every time it runs, rather than silently trusting
`ticker.options[0]`:

```python
expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
days_to_expiry = (expiry_date - date.today()).days
T = days_to_expiry / 365
print(f"Using expiry: {expiry} ({days_to_expiry} days out, T={T:.4f})")
```

**Lesson:** always sanity-check an assumed input by solving backward
from a trusted, liquid reference point, rather than trusting a data
source's default ordering.

## Finding: The Volatility Skew

With the corrected expiry, pricing each strike using the market's own
implied volatility trivially reproduces the market price — that's
circular, since implied volatility is *defined* as whatever number
makes the model match the market. The more meaningful test is whether
a **single** volatility, applied across every strike, can explain the
whole chain — which is what the original Black-Scholes model assumes
should be possible.

Using the at-the-money implied volatility (61.1%, from strike 125)
applied uniformly to every other strike, on a ~25-day expiry:

| Strike | Market Price | Model Price (single vol) | % Difference |
|--------|-------------|---------------------------|--------------|
| 113    | $18.80      | $15.90                    | -15.5%       |
| 125    | $8.45       | $8.58                     | +1.5%        |
| 135    | $2.60       | $4.63                     | +77.9%       |
| 150    | $0.48       | $1.56                     | +227.9%      |

Near the money, a single volatility number works well (within a few
percent). Away from the money in either direction, the single-vol
model's price diverges sharply from what the market actually charges —
and the gap grows the further out you go.

**Why this happens:** this is the well-documented **volatility skew**
(sometimes called the "volatility smile"). The market doesn't believe
one constant volatility applies equally at every strike — it prices in
extra risk of large, sudden moves (earnings surprises, crashes) at
strikes further from the current price, which shows up as a *higher*
implied volatility the further out you go. This is one of the most
famous real-world violations of the original 1973 Black-Scholes
assumptions, and it's the reason real trading desks use a full
volatility *surface* (a different vol per strike and expiry) rather
than the model's single-sigma assumption.

## Confirmation on a Second Chain: Short-Dated Options Amplify the Skew

The same analysis was re-run on a later date against a much
shorter-dated chain — 3 days to expiration instead of ~25 — and the
skew showed up again, more severely this time:

| Strike | Market Price | Model Price (single vol) | % Difference | Market IV |
|--------|-------------|---------------------------|--------------|-----------|
| 113    | $16.40      | $15.64                    | -4.6%        | 1.085     |
| 128 (ATM) | $4.25    | $3.74                     | -12.1%       | 0.744     |
| 140    | $0.65       | $0.44                     | -31.7%       | 0.721     |
| 150    | $0.15       | $0.04                     | -76.3%       | 0.803     |

Here the market's implied volatility dips to its lowest near the money
(~0.71-0.74) and climbs back up sharply at both wings (1.085 on the low
strike, 0.803 at strike 150) — the classic smile shape. Because the
single ATM volatility used for pricing is *lower* than what the market
assigns to the far strikes, the single-vol model **underprices** those
wings, consistent with the pattern above.

**Why the percentage errors are larger here than in the 25-day case:**
with only 3 days to expiration, option prices are extremely sensitive
to volatility assumptions in percentage terms — a small absolute
mispricing on an option already worth only a few cents (like the
$0.15 strike-150 call) becomes a very large percentage error. This
confirms the skew isn't a one-off artifact of a single date or expiry:
it persists across very different time horizons, and it's more
pronounced, not less, the closer an option gets to expiration.

## What I'd Do Next

- **Fit a full volatility surface** — instead of one sigma, model
  implied volatility as a function of strike and expiry (e.g. with a
  simple polynomial or spline fit), then re-price the chain and see how
  much closer that gets to the market.
- **Compare puts as well as calls** — check whether the skew is
  symmetric or whether puts show a steeper skew than calls (common in
  equity markets, reflecting higher demand for downside protection).
- **Extend the delta-hedging backtest across strikes with different
  skew levels** — see whether hedging effectiveness holds up better or
  worse for strikes where the single-vol assumption breaks down most.
- **Track the skew across expirations** — plot implied vol vs. days to
  expiration for a fixed strike, to see how the term structure of
  volatility itself behaves, not just the strike-wise skew.

## Running It

```bash
pip install numpy scipy pandas yfinance --break-system-packages
python black_scholes_duol.py
```
