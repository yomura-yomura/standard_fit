import numpy as np


__all__ = [
    "gaussian", "power_law", "exp", "log",
    "sin",
    "exp10",
    "tanh",
    "approx_landau",
    "na_pol1"
]


def na_pol1(x, p0, p1):
    return p0 + p1 * x


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


__all__ += ["log_gaussian", "log_gaussian2"]


def log_gaussian(x, A, μ, σ):
    """
    log-normal distribution:
        μ: mean on the logarithm scale
        σ: standard deviation on the logarithm scale
    """
    return A / (np.sqrt(2 * np.pi) * σ * x) * np.exp(-0.5 * ((np.log(x) - μ) / σ) ** 2)


def log_gaussian2(x, A, μ, σ):
    """
    Alternative parameterization of log-normal distribution:
        μ: mean on the natural scale
        σ: standard deviation on the natural scale
    """
    term = 1 + (σ / μ) ** 2
    log_term = np.log(term)
    return A / (x * np.sqrt(2 * np.pi * log_term)) * np.exp(-0.5 * np.log(x / (μ / np.sqrt(term))) ** 2 / log_term)

