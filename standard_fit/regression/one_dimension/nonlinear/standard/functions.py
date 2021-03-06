import numpy as np


__all__ = [
    "gaussian", "power_law", "exp", "log",
    "sin",
    "exp10",
    "tanh",
    "approx_landau",
    "sqrt",
    *(f"na_pol{i}" for i in range(10))
]


def sqrt(x, a, b, c):
    return np.sqrt((x - c) / a) + b


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


__all__ += ["log_gaussian"]


def log_gaussian(x, A, μ, σ):
    """
    log-normal distribution:
        μ: mean on the logarithm scale
        σ: standard deviation on the logarithm scale
    """
    # return A / (np.sqrt(2 * np.pi) * σ * x) * np.exp(-0.5 * ((np.log(x) - μ) / σ) ** 2)
    return gaussian(np.log(x), A, np.log(μ), np.log(σ))
#
# # Alternative parameterization of log-normal distribution
# # See https://en.wikipedia.org/wiki/Log-normal_distribution#Alternative_parameterizations
#
# __all__ += ["log_gaussian2"]
#
# def log_gaussian2(x, A, μ, σ):
#     """
#     Alternative parameterization of log-normal distribution:
#         μ: mean on the natural scale
#         σ: standard deviation on the natural scale
#     """
#     term = 1 + (σ / μ) ** 2
#     log_term = np.log(term)
#     return A / (x * np.sqrt(2 * np.pi * log_term)) * np.exp(-0.5 * np.log(x / (μ / np.sqrt(term))) ** 2 / log_term)
#
#
# __all__ += ["log10_gaussian", "log10_gaussian2"]
#
#
# def log10_gaussian(x, A, μ, σ):
#     """
#     log-normal distribution:
#         μ: mean on the logarithm scale
#         σ: standard deviation on the logarithm scale
#     """
#     return gaussian(np.log10(x), A, np.log10(μ), np.log10(σ))
#
#
# def log10_gaussian2(x, A, μ, σ):
#     """
#     log-normal distribution:
#         μ: mean on the normal_scale
#         σ: standard deviation on the normal scale
#     """
#     term = 1 + (σ / μ) ** 2
#     log_term = np.log10(term)
#     return A / (x * np.sqrt(2 * np.pi * log_term)) * np.exp(-0.5 * np.log10(x / (μ / np.sqrt(term))) ** 2 / log_term)


def _make_pol_n(n):
    # Make n-dimensional polynomial
    kwargs = [f"p{i}" for i in range(n + 1)]
    formula = "+".join([f"{p}*x**{i}" for i, p in enumerate(kwargs)])
    unpacked_kwargs = ", ".join(kwargs)
    exec(f"def na_pol{n}(x, {unpacked_kwargs}): return {formula}")
    globals()[f"na_pol{n}"] = eval(f"na_pol{n}")


for i in range(10):
    _make_pol_n(i)
