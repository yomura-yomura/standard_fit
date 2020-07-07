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


def fit(x, y, fit_type, y_err=None, initial_guess=None, bounds=None):
    assert len(x) == len(y)
    if fit_type not in fit_list:
        raise ValueError(f"Not available fit type: {fit_type}")

    x = np.array(x)
    y = np.array(y)

    invalid = np.unique(
        (
            [i for i, (isnan, isinf) in enumerate(zip(np.isnan(x), np.isinf(x))) if (isnan or isinf)]
            if np.issubdtype(x.dtype, np.floating) else []
        ) +
        (
            [i for i, (isnan, isinf) in enumerate(zip(np.isnan(y), np.isinf(y))) if (isnan or isinf)]
            if np.issubdtype(y.dtype, np.floating) else []
        )
    ).astype(int)

    x = np.delete(x, invalid)
    y = np.delete(y, invalid)
    if y_err is not None:
        assert len(x) == len(y_err)
        y_err = np.delete(y_err, invalid)
    else:
        y_err = np.array([1] * len(x))

    # if fit_type == "kde":
    #     import scipy.stats
    #     func = scipy.stats.gaussian_kde(x)
    #
    #     import iminuit
    #     result = iminuit.minimize(
    #         lambda p: -func(*p), x0=(x[np.argmax(y)],)
    #     )
    #     x0 = result["x"]
    #     pk = -result["fun"]
    #
    #     cov_params = np.full((3, 3), np.nan)
    #     chi_squared = np.sum((y - func(x))**2) / len(x)
    #     return fit_type, (func, pk, x0), cov_params, chi_squared, None

    if initial_guess is None:
        if fit_type == "gaussian":
            a = np.max(y)
            mean = np.average(x, weights=y)
            rms = np.average((x - mean)**2, weights=y)
            initial_guess = (a, mean, rms)
            # bounds = (
            #     [0, -np.inf, 0],
            #     np.inf
            # )
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

    fit_func = get_func(fit_type)

    # import scipy.optimize
    # params, cov_params = scipy.optimize.curve_fit(fit_func, x, y, sigma=y_err, p0=initial_guess, bounds=bounds)
    import iminuit
    result = iminuit.minimize(
        lambda p: np.sum(((fit_func(x, *p) - y)/y_err)**2),
        x0=initial_guess, bounds=bounds
    )
    params, cov_params = result["x"], result["hess_inv"]

    chi_squared = np.sum((fit_func(x, *params) - y) ** 2)/len(x)
    ndf = len(x) - len(params)
    return fit_type, params, cov_params, chi_squared, ndf


def get_func(fit_type):
    fit_type = fit_type.replace(" ", "_")
    return globals()[fit_type]


def get_parameter_names(fit_type):
    fit_func = get_func(fit_type)
    import inspect
    return list(inspect.signature(fit_func).parameters)[1:]


def add_annotation(fig, fit_results):
    fit_type, params, cov_params, chi_squared, ndf = fit_results

    err_params = [np.sqrt(cov_params[i][i]) for i in range(len(params))]

    def as_str(a, formats=None, align="right"):
        if formats is None:
            a = [f"{s}" for s in a]
        else:
            a = [f"{s:{f}}" for s, f in zip(a, formats)]
        max_len = max([len(s) for s in a])
        if align == "right":
            return [f"{s:>{max_len}}" for s in a]

    from uncertainties import ufloat
    if fit_type == "kde":
        str_params = [f"{ufloat(p, ep):.4g}".replace("+/-", " ± ") for p, ep in zip(params[1:], err_params[1:])]
        text_lines = [
            f"χ²/ndf = {chi_squared:.4g}/{ndf}",
            *[f"{n} = {p}" for n, p in zip(as_str(get_parameter_names(fit_type)[1:]), str_params)]
        ]
    else:
        str_params = [f"{ufloat(p, ep):.4g}".replace("+/-", " ± ") for p, ep in zip(params, err_params)]

        text_lines = [
            f"χ²/ndf = {chi_squared:.4g}/{ndf}",
            *[f"{n} = {p}" for n, p in zip(as_str(get_parameter_names(fit_type)), str_params)]
        ]

    fig.add_annotation(
        # x=x0,
        # y=y0,
        x=1,
        y=0.83,
        xref="paper",
        yref="paper",

        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        align="left",
        showarrow=False,
        text="<br>".join(text_lines)
    )


# Gaussian Fit Test
if __name__ == "__main__" and False:
    x = np.random.normal(loc=5, scale=1.5, size=10000)
    y, bins = np.histogram(x, bins="auto")
    x = (bins[:-1] + bins[1:])/2
    result = fit(x, y, "gaussian")

    print("gaussian fit results:")
    for name, param in zip(get_parameter_names("gaussian"), result[1]):
        print(f"\t{name} = {param}")

    fit_x = np.linspace(x.min(), x.max(), 100)
    fit_y = [gaussian(x, *result[1]) for x in fit_x]

    import plotly.graph_objs as go
    fig = go.Figure(
        data=[
            go.Scatter(name="data", mode="markers", x=x, y=y),
            go.Scatter(name="fit", mode="lines", x=fit_x, y=fit_y)
        ]
    )

    add_annotation(fig, result)
    fig.show()

if __name__ == "__main__":
    x = np.linspace(-5, 5, 20)
    y = np.poly1d([1, 3, -9, 4])(x) + np.random.normal(0, 2, size=len(x))
    result = fit(x, y, "pol3")

    print("pol3 fit results:")
    for name, param in zip(get_parameter_names("pol3"), result[1]):
        print(f"\t{name} = {param}")

    fit_x = np.linspace(x.min(), x.max(), 100)
    fit_y = [pol3(x, *result[1]) for x in fit_x]

    import plotly.graph_objs as go

    fig = go.Figure(
        data=[
            go.Scatter(name="data", mode="markers", x=x, y=y),
            go.Scatter(name="fit", mode="lines", x=fit_x, y=fit_y)
        ]
    )

    add_annotation(fig, result)
    fig.show()

