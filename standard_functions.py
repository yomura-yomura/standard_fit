import numpy as np


fit_list = [
    "gaussian", "power law", "exp", "log",
    "exp10",
    "tanh",
    "approx_landau", "kde",
    *[f"pol{i}" for i in range(4)]
]


def gaussian(x, A, μ, σ):
    return A * np.exp(-0.5 * ((x - μ) / σ) ** 2)


def power_law(x, A, s):
    return A * np.power(x, s)


def exp(x, p0, p1):
    return p0 * np.exp(p1 * x)


def log(x, p0, p1):
    return p0 * np.log(p1 * x)


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


# Maybe will be removed
def kde(x, _func, pk, x0):
    return _func(x)


# Make n-dimensional polynomial


def _make_pol_n(n):
    kwargs = [f"p{i}" for i in range(n + 1)]
    formula = "+".join([f"{p}*x**{i}" for i, p in enumerate(kwargs)])
    unpacked_kwargs = ", ".join(kwargs)
    exec(f"def pol{n}(x, {unpacked_kwargs}): return {formula}")
    func = eval(f"pol{n}")
    globals()[f"pol{n}"] = func
    return func


for i in range(4):
    _make_pol_n(i)


# Utilities


def get_func(fit_type):
    fit_type = fit_type.replace(" ", "_")
    return globals()[fit_type]


def get_parameter_names(fit_type):
    fit_func = get_func(fit_type)
    import inspect
    return list(inspect.signature(fit_func).parameters)[1:]


def estimate_initial_guess(fit_type, x, y):
    if fit_type == "gaussian":
        a = np.max(y)
        # mean = np.average(x, weights=y)
        mean = x[np.argmax(y)]
        rms = np.sqrt(np.average((x - mean) ** 2, weights=y))
        initial_guess = (a, mean, rms)
        bounds = (
            [0, -np.inf, 0],
            np.inf
        )
    elif fit_type == "power law":
        min_i = np.argmin(x)
        max_i = np.argmax(x)
        s = (np.log(y[max_i]) - np.log(y[min_i])) / (np.log(x[max_i]) - np.log(x[min_i]))
        a = y[min_i] * np.power(x[min_i], -s)
        initial_guess = (a, s)
    elif fit_type.startswith("pol"):
        n_pol = int(fit_type[3:]) + 1
        sampled_x = np.linspace(min(x), max(x), n_pol)
        sampled_y = np.take(y, np.searchsorted(x, sampled_x) - 1)
        initial_guess = sampled_y @ np.linalg.inv([[x ** i for i in range(n_pol)] for x in sampled_x])
    elif fit_type == "exp":
        min_i = np.argmin(x)
        max_i = np.argmax(x)
        p1 = (np.log(y[max_i]) - np.log(y[min_i])) / (x[max_i] - x[min_i])
        med_i = np.searchsorted(x, np.mean(x))
        p0 = y[med_i] * np.exp(-p1 * x[med_i])
        initial_guess = (p0, p1)
    elif fit_type == "log":
        min_i = np.argmin(x)
        max_i = np.argmax(x)
        p1 = (np.log(x[max_i]) - np.log(x[min_i])) / (y[max_i] - y[min_i])
        p0 = x[min_i] * np.exp(-p1 * y[min_i])
        initial_guess = (p0, p1)
    elif fit_type == "exp10":
        min_i = np.argmin(x)
        max_i = np.argmax(x)
        p1 = (np.log10(y[max_i]) - np.log10(y[min_i])) / (x[max_i] - x[min_i])
        med_i = np.searchsorted(x, np.mean(x))
        p0 = y[med_i] * np.power(10, -p1 * x[med_i])
        initial_guess = (p0, p1)
    elif fit_type == "tanh":
        C = (np.max(y) - np.min(y)) * 0.5
        y0 = (np.min(y) + np.max(y)) * 0.5
        x0 = x[np.searchsorted(y, y0)]
        a = 1
        initial_guess = (C, a, x0, y0)
    elif fit_type == "approx_landau":
        max_i = np.argmax(y)
        A = y[max_i]
        m = x[max_i]
        s = A * 1
        initial_guess = (A, m, s)
    else:
        raise NotImplementedError

    # print(initial_guess)
    return initial_guess
