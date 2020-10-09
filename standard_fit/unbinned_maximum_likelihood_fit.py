import numpy as np
from . import standard_functions


def integrate(func, params, min_x, max_x):
    import scipy.integrate
    return scipy.integrate.quad(lambda x: func(x, *params), min_x, max_x)


def get_normalized_function(fit_type, min_x, max_x):
    func = standard_functions.get_func(fit_type)

    def norm_func(x, *params):
        return func(x, *params) / (integrate(func, params, min_x, max_x)[0])

    return norm_func


def fit(x, fit_type, x_range=()):
    if len(x_range) == 0:
        x_range = (x.min(), x.max())
    elif len(x_range) == 2:
        x = x[(x_range[0] < x) & (x < x_range[1])]
    else:
        raise ValueError

    norm_func = get_normalized_function(fit_type, *x_range)

    def fcn(params):
        p = norm_func(x, *params)
        lnL = np.sum(np.log(p[p != 0]))
        # print(lnL)
        return -lnL

    if fit_type == "gaussian":
        x0 = (1, 0, 1)
        bounds = [(1, 1), None, None]
    else:
        import warnings
        warnings.warn(f"Unspecified fit_type = {fit_type}")
        x0 = [1] * len(standard_functions.get_parameter_names(fit_type))
        bounds = None

    import iminuit
    result = iminuit.minimize(fcn, x0, bounds=bounds)
    params, cov_params = result["x"], result["hess_inv"]

    if fit_type == "gaussian":
        params[0] *= norm_func(params[1], *params)

    chi_squared = np.nan  # not implemented yet
    ndf = len(x) - len(params)
    return fit_type, params, cov_params, chi_squared, ndf, x_range
