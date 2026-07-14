"""
black_scholes_duol.py
-----------------------
Black-Scholes model applied to a real Duolingo (DUOL) options chain,
including the expiry-verification fix and the volatility skew analysis
documented in the project README.
"""

import numpy as np
from scipy.stats import norm  # type: ignore
from dataclasses import dataclass
from enum import Enum
from datetime import date, datetime
import yfinance as yf  # type: ignore


# ----------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------
class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionInputs:
    S: float      # Current underlying price
    K: float      # Strike price
    T: float      # Time to expiration (years)
    r: float      # Risk-free rate (decimal)
    sigma: float  # Volatility (decimal)

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError("Current underlying price (S) must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price (K) must be positive.")
        if self.T <= 0:
            raise ValueError("Time to expiration (T) must be positive.")
        if self.sigma < 0:
            raise ValueError("Volatility (sigma) cannot be negative.")


class BlackScholes:
    def __init__(self, inputs: OptionInputs):
        self.inputs = inputs
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self) -> tuple[float, float]:
        S, K, T, r, sigma = (
            self.inputs.S, self.inputs.K, self.inputs.T,
            self.inputs.r, self.inputs.sigma,
        )
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price(self, option_type: OptionType) -> float:
        S, K, T, r = (self.inputs.S, self.inputs.K, self.inputs.T, self.inputs.r)
        d1, d2 = self._d1, self._d2

        if option_type == OptionType.CALL:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == OptionType.PUT:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def delta(self, option_type: OptionType) -> float:
        if option_type == OptionType.CALL:
            return norm.cdf(self._d1)
        return norm.cdf(self._d1) - 1

    def gamma(self) -> float:
        S, T, sigma = self.inputs.S, self.inputs.T, self.inputs.sigma
        return norm.pdf(self._d1) / (S * sigma * np.sqrt(T))

    def vega(self) -> float:
        S, T = self.inputs.S, self.inputs.T
        return S * norm.pdf(self._d1) * np.sqrt(T)

    def theta(self, option_type: OptionType) -> float:
        S, K, T, r, sigma = (
            self.inputs.S, self.inputs.K, self.inputs.T,
            self.inputs.r, self.inputs.sigma,
        )
        d1, d2 = self._d1, self._d2
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            return term1 + term2
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            return term1 + term2

    def rho(self, option_type: OptionType) -> float:
        K, T, r = self.inputs.K, self.inputs.T, self.inputs.r
        d2 = self._d2

        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ----------------------------------------------------------------------
# Data pipeline: pull a real, liquidity-filtered DUOL chain
# ----------------------------------------------------------------------
def get_liquid_chain(symbol: str = "DUOL"):
    """
    Pulls the nearest options chain for `symbol`, verifies the actual
    days-to-expiration (this is the exact check that caught the original
    expiry bug), and filters out illiquid/stale strikes.
    """
    ticker = yf.Ticker(symbol)

    expiry = ticker.options[0]
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    today = date.today()
    days_to_expiry = (expiry_date - today).days
    T = days_to_expiry / 365

    print(f"Using expiry: {expiry} ({days_to_expiry} days out, T={T:.4f})")

    chain = ticker.option_chain(expiry).calls

    liquid = chain[
        (chain["volume"] > 10) &
        (chain["openInterest"] > 50) &
        (chain["bid"] > 0)
    ].copy()

    liquid["mid_price"] = (liquid["bid"] + liquid["ask"]) / 2

    current_price = ticker.history(period="1d")["Close"].iloc[-1]

    return liquid, current_price, T


# ----------------------------------------------------------------------
# Analysis: single ATM volatility vs. the full chain (the skew finding)
# ----------------------------------------------------------------------
def analyze_skew(liquid_chain, S: float, T: float, r: float = 0.038):
    """
    Prices every liquid strike using a SINGLE volatility (the at-the-money
    implied vol), then compares against real market prices. This is the
    genuine test of the Black-Scholes single-volatility assumption — see
    README.md for why using each strike's own implied vol would be
    circular.
    """
    atm_idx = (liquid_chain["strike"] - S).abs().idxmin()
    atm_sigma = liquid_chain.loc[atm_idx, "impliedVolatility"]
    print(f"\nUsing single ATM volatility from strike "
          f"{liquid_chain.loc[atm_idx, 'strike']}: {atm_sigma:.4f}\n")

    results = []
    for _, row in liquid_chain.iterrows():
        inputs = OptionInputs(S=S, K=row["strike"], T=T, r=r, sigma=atm_sigma)
        bs = BlackScholes(inputs)
        model_price = bs.price(OptionType.CALL)
        diff_pct = (model_price - row["mid_price"]) / row["mid_price"] * 100

        results.append({
            "strike": row["strike"],
            "market_mid": row["mid_price"],
            "market_iv": row["impliedVolatility"],
            "model_price_single_vol": model_price,
            "diff_pct": diff_pct,
        })

    return results


def print_results(results):
    print(f"{'Strike':>7} {'Market':>9} {'Model':>9} {'% Diff':>9} {'Mkt IV':>8}")
    for r in results:
        print(f"{r['strike']:>7.1f} {r['market_mid']:>9.3f} "
              f"{r['model_price_single_vol']:>9.3f} {r['diff_pct']:>8.1f}% "
              f"{r['market_iv']:>8.3f}")


if __name__ == "__main__":
    liquid_chain, S, T = get_liquid_chain("DUOL")

    print(f"\nCurrent DUOL price: {S:.2f}")
    print(liquid_chain[["strike", "bid", "ask", "mid_price",
                        "volume", "openInterest", "impliedVolatility"]])

    results = analyze_skew(liquid_chain, S=S, T=T)
    print_results(results)
