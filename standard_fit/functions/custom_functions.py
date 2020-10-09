import numpy as np
from . import standard_functions


__all__ = ["spectrum", "spectrum_linear", "broken_pol1"]


@np.vectorize
def broken_power_law(x, A, x0, s1, s2):
    # s1 = -1.6 - 1
    # s2 = -3.3 - 1
    if x < x0:
        return standard_functions.power_law(x, A * np.power(x0, -s1), s1)
    else:
        return standard_functions.power_law(x, A * np.power(x0, -s2), s2)


@np.vectorize
def _inner_broken_pol1(x, A, bp, s1, s2):
    if x < bp:
        return standard_functions.pol1(x, A - s1*bp, s1)
    else:
        return standard_functions.pol1(x, A - s2*bp, s2)


def broken_pol1(x, A, bp, s1, s2):
    # s1 = -1.65
    # s2 = -3.28
    return _inner_broken_pol1(x, A, bp, s1, s2)


def spectrum(x, a, x0, A, bp, s2):
    return standard_functions.tanh(x, 0.5, a, x0, 0.5) * broken_power_law(x, A, bp, s2)


def spectrum_linear(x, a, x0, A, bp, s2):
    return standard_functions.tanh(x, 0.5, a, x0, 0.5) * broken_pol1(x, A, bp, s2)


# Estimate initial guess


def estimate_initial_guess(fit_type, x, y):
    if fit_type == "spectrum":
        a = 1e-1
        # a = 3
        x0 = (x[np.argmax(y)] + x[np.argmin(y)]) * 0.5
        # s = -3
        bp = x[np.argmax(y)]
        A = np.max(y)
        s2 = -3.28
        # s2 = -160
        initial_guess = (a, x0, A, bp, s2)
    elif fit_type == "spectrum_linear":
        a = 1
        x0 = (x[np.argmax(y)] + x[np.argmin(y)]) * 0.5
        bp = x[np.argmax(y)]
        A = np.max(y)
        s2 = -3.28
        initial_guess = (a, x0, A, bp, s2)
    elif fit_type == "broken_pol1":
        bp = (np.min(x) + np.max(x)) * 0.5
        A = y[np.searchsorted(x, bp)]
        s1 = (A - np.min(y)) / (bp - np.min(x))
        s2 = (np.max(y) - A) / (np.max(x) - bp)
        initial_guess = (A, bp, s1, s2)
    else:
        raise NotImplementedError
    return initial_guess

