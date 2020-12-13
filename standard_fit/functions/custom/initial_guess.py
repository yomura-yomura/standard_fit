import numpy as np


__all__ = ["estimate_initial_guess"]


def spectrum(x, y):
    a = 1e-1
    # a = 3
    x0 = (x[np.argmax(y)] + x[np.argmin(y)]) * 0.5
    # s = -3
    bp = x[np.argmax(y)]
    A = np.max(y)
    s2 = -3.28
    # s2 = -160
    return a, x0, A, bp, s2


def spectrum_linear(x, y):
    a = 1
    x0 = (x[np.argmax(y)] + x[np.argmin(y)]) * 0.5
    bp = x[np.argmax(y)]
    A = np.max(y)
    s2 = -3.28
    return a, x0, A, bp, s2


def broken_pol1(x, y):
    bp = (np.min(x) + np.max(x)) * 0.5
    A = y[np.searchsorted(x, bp)]
    s1 = (A - np.min(y)) / (bp - np.min(x))
    s2 = (np.max(y) - A) / (np.max(x) - bp)
    return A, bp, s1, s2


def estimate_initial_guess(fit_type, x, y):
    if fit_type in globals():
        return globals()[fit_type](x, y)
    else:
        raise NotImplementedError(fit_type)
