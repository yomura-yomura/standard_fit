import numpy as np
from ..standard import functions


__all__ = ["spectrum", "spectrum_linear", "broken_pol1"]


@np.vectorize
def broken_power_law(x, A, x0, s1, s2):
    # s1 = -1.6 - 1
    # s2 = -3.3 - 1
    if x < x0:
        return functions.power_law(x, A * np.power(x0, -s1), s1)
    else:
        return functions.power_law(x, A * np.power(x0, -s2), s2)


@np.vectorize
def _inner_broken_pol1(x, A, bp, s1, s2):
    if x < bp:
        return functions.pol1(x, A - s1*bp, s1)
    else:
        return functions.pol1(x, A - s2*bp, s2)


def broken_pol1(x, A, bp, s1, s2):
    # s1 = -1.65
    # s2 = -3.28
    return _inner_broken_pol1(x, A, bp, s1, s2)


def spectrum(x, a, x0, A, bp, s2):
    return functions.tanh(x, 0.5, a, x0, 0.5) * broken_power_law(x, A, bp, s2)


def spectrum_linear(x, a, x0, A, bp, s2):
    return functions.tanh(x, 0.5, a, x0, 0.5) * broken_pol1(x, A, bp, s2)


def _get_func(fit_type: str):
    assert fit_type in __all__
    return globals()[fit_type]

