import numpy as np


__all__ = [
    "gaussian", "power_law", "exp", "log",
    "sin",
    "exp10",
    "tanh",
    "approx_landau"
]


def gaussian(x, A, μ, σ):
    return A / (np.sqrt(2 * np.pi) * σ) * np.exp(-0.5 * ((x - μ) / σ) ** 2)


def power_law(x, A, s):
    return A * np.power(x, s)


def exp(x, p0, p1):
    return p0 * np.exp(p1 * x)


def log(x, p0, p1):
    return p0 * np.log(p1 * x)


def sin(x, A, ω, x0):
    return A * np.sin(ω * (x - x0))


def exp10(x, p0, p1):
    return p0 * np.power(10, p1 * x)


def tanh(x, C, a, x0, y0):
    return C * np.tanh(a * (x - x0)) + y0


def approx_landau(x, A, m, s):
    """
    :param x:
    :param A: peak
    :param m: mode
    :param s: scale parameter
    :return:
    """

    t = (x - m) / s
    return A * np.exp(-0.5 * (t + np.exp(-t)) + 0.5)


def get_func(fit_type: str):
    assert fit_type in __all__
    return globals()[fit_type]


