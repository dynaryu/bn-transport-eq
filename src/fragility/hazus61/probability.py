"""Probability operations with strict finite-domain and state-order validation."""
import numpy as np
from scipy.special import ndtr


def nonnegative(value, label):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def lognormal_exceedance(im, median, beta):
    im = nonnegative(im, "IM")
    median, beta = np.asarray(median, dtype=float), np.asarray(beta, dtype=float)
    if not np.isfinite(median).all() or not np.isfinite(beta).all() or (median <= 0).any() or (beta <= 0).any():
        raise ValueError("median and beta must be finite and positive")
    with np.errstate(divide="ignore"):
        return ndtr((np.log(im) - np.log(median)) / beta)


def validate_exceedance(q):
    q = np.asarray(q, dtype=float)
    if q.ndim < 1 or q.shape[-1] < 1 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or (np.diff(q, axis=-1) > 0).any():
        raise ValueError("non_monotone_exceedance: require 1 >= qS >= ... >= qC >= 0")
    return q


def cumulative_to_discrete(q):
    q = validate_exceedance(q)
    return np.concatenate([1 - q[..., :1], -np.diff(q, axis=-1), q[..., -1:]], axis=-1)


def combine_independent_exceedance(gs, gf):
    gs, gf = validate_exceedance(gs), validate_exceedance(gf)
    if gs.shape != gf.shape:
        raise ValueError("hazard modes must have matching states and batch shapes")
    return validate_exceedance(1 - (1 - gs) * (1 - gf))
