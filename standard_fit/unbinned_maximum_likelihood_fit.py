import numpy as np
from . import regression
# from .regression import nonlinear
import scipy.integrate
import warnings

try:
    import IPython.display
    display = IPython.display.display
except ImportError:
    display = print


def integrate(func, params, min_x, max_x):
    return scipy.integrate.quad(lambda x: func(x, *params), min_x, max_x)


def get_normalized_function(fit_type, min_x, max_x):
    is_multivariate = False
    nonlinear_module = regression.multi_dimension.nonlinear if is_multivariate else regression.one_dimension.nonlinear

    func = nonlinear_module.get_func(fit_type)

    def norm_func(x, *params):
        return func(x, *params) / (integrate(func, params, min_x, max_x)[0])

    return norm_func


def fit(x, fit_type, x_range=(), print_result=True):
    is_multivariate = False

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
        return -lnL

    if fit_type == "gaussian":
        x0 = (1, 0, 1)
        bounds = [(1, 1), None, None]
    else:
        warnings.warn(f"Unspecified fit_type = {fit_type}")
        x0 = [1] * len(regression.get_parameter_names(fit_type, is_multivariate))
        bounds = None

    import iminuit
    m = iminuit.Minuit(fcn, x0)
    m.limits = bounds
    m.errordef = iminuit.Minuit.LIKELIHOOD
    m.strategy = 2
    m.simplex()
    m.migrad()

    params = m.values

    # if fit_type == "gaussian":
    #     params[0] /= norm_func(params[1], *params)

    y_range = (-np.inf, np.inf)  # Not implemented yet

    if print_result:
        display(m.fmin)
        display(m.params)

    return (
        fit_type, tuple(params), tuple(m.errors), m.fval/2, len(x) - m.nfit,
        x_range, y_range, is_multivariate
    )
