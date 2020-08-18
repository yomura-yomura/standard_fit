import numpy as np
from . import functions


def fit(x, y, fit_type, y_err=None, initial_guess=None, bounds=None, x_range=(), print_result=True):
    assert len(x) == len(y)
    if fit_type not in functions.fit_list:
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

    if len(x_range) == 0:
        x_range = (np.min(x), np.max(x))
    elif len(x_range) == 2:
        x_range = list(x_range)
        if x_range[0] is None:
            x_range[0] = x.min()
        if x_range[1] is None:
            x_range[1] = x.max()

        mask = (x_range[0] <= x) & (x <= x_range[1])
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            raise ValueError

    else:
        raise ValueError

    if y_err is not None:
        assert len(x) == len(y_err)
        y_err = np.delete(y_err, invalid)
    else:
        y_err = np.array([1] * len(x))

    if initial_guess is None:
        initial_guess = functions.estimate_initial_guess(fit_type, x, y)
        # print(f"{initial_guess = }")

    fit_func = functions.get_func(fit_type)

    # import scipy.optimize
    # params, cov_params = scipy.optimize.curve_fit(fit_func, x, y, sigma=y_err, p0=initial_guess, bounds=bounds)
    import iminuit
    result = iminuit.minimize(
        lambda p: np.sum(((fit_func(x, *p) - y)/y_err)**2),
        x0=initial_guess, bounds=bounds
    )
    params, cov_params = result["x"], result["hess_inv"]

    # sqrt(cov_params) is a half length and divided by sqrt(number of points)
    cov_params /= 4 * len(x)

    chi_squared = np.sum((fit_func(x, *params) - y) ** 2)
    ndf = len(x) - len(params)

    if print_result:
        print(result)
#        print(f"{fit_type} fit results:")
#        for name, param in zip(functions.get_parameter_names(fit_type), params):
#            print(f"\t{name} = {param}")

    return fit_type, params, cov_params, chi_squared, ndf, x_range


def gaussian_fit(x, **kwargs):
    counts, bins = np.histogram(x, bins="auto")
    x = (bins[1:] + bins[:-1]) * 0.5
    y = counts

    return fit(x, y, "gaussian", **kwargs)


def gaussian_fit_and_show(x, **kwargs):
    counts, bins = np.histogram(x, bins="auto")
    result = fit((bins[1:] + bins[:-1]) * 0.5, counts, "gaussian", **kwargs)

    from . import _plotly_express as _px
    from . import _plotly
    fig = _px.histogram(x, bins=bins)
    fig.add_trace(_plotly.get_fit_trace(result))
    _plotly.add_annotation(fig, result)

    # import plotly
    # plotly.offline.plot(fig, config=dict(editable=True))
    fig.show(config=dict(editable=True))


def fit_and_fig(x, y, fit_type, px_kwargs={}, *args, **kwargs):
    from . import _plotly_express as _px
    result = fit(x, y, fit_type, *args, **kwargs)
    fig = _px.scatter(x, y, result, **px_kwargs)
    return fig


def fit_and_show(x, y, fit_type, px_kwargs={}, *args, **kwargs):
    from . import _plotly_express as _px
    result = fit(x, y, fit_type, *args, **kwargs)
    fig = _px.scatter(x, y, result, **px_kwargs)
    fig.show(config=dict(editable=True))
    




