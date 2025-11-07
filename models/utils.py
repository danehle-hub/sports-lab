# models/utils.py (FULL REPLACEMENT - vectorized & robust)

import numpy as np
import pandas as pd
from math import erf, sqrt

EV_STRINGS = {"ev", "even", "pk", "pick", "pickem", "pick’em", "pick'em"}

def _american_to_decimal_one(x) -> float:
    """Parse a single American-odds value to decimal odds."""
    if x is None:
        return np.nan
    s = str(x).strip().lower().replace(",", "")
    if s in EV_STRINGS:
        return 2.0
    try:
        a = float(s)
    except Exception:
        return np.nan
    if a >= 100:
        return 1.0 + (a / 100.0)
    if a <= -100:
        return 1.0 + (100.0 / abs(a))
    return np.nan

def american_to_decimal(price):
    """
    Vectorized: accepts scalar, list, np.array, or pd.Series.
    Returns same 'shape' (pd.Series in -> pd.Series out; otherwise np.ndarray or float).
    """
    if np.isscalar(price):
        return _american_to_decimal_one(price)
    arr = np.asarray(price, dtype=object)
    vec = np.vectorize(_american_to_decimal_one, otypes=[float])(arr)
    # Preserve Pandas Series if that was the input
    if isinstance(price, pd.Series):
        return pd.Series(vec, index=price.index, name=getattr(price, "name", None))
    return vec

def _ev_one(p: float, dec: float) -> float:
    """EV per $1 risked for a single (p, dec) pair."""
    if p is None:
        return np.nan
    try:
        p = float(p)
    except Exception:
        return np.nan
    if not (0.0 <= p <= 1.0) or dec is None or not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    b = dec - 1.0
    return p * b - (1.0 - p)

def ev_from_prob_and_american(prob, american):
    """
    Vectorized EV per $1 risked.
    prob: scalar/array/Series of win probabilities (0..1)
    american: scalar/array/Series of American odds (e.g., -110, +150, 'EV', etc.)
    """
    dec = american_to_decimal(american)
    # Broadcast to common shape
    p_arr = np.asarray(prob, dtype=float)
    d_arr = np.asarray(dec, dtype=float)
    # Elementwise EV
    # Use vectorize to keep NaNs where appropriate
    f = np.vectorize(_ev_one, otypes=[float])
    ev = f(p_arr, d_arr)
    if isinstance(prob, pd.Series):
        return pd.Series(ev, index=prob.index, name="ev")
    return ev

def norm_cdf_np(x):
    """
    Vectorized normal CDF using math.erf for compatibility (NumPy does not expose erf).
    """
    x = np.asarray(x, dtype=float)
    # vectorize math.erf to handle arrays
    erf_vec = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(x / sqrt(2.0)))
