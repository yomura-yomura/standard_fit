# Maybe will be removed
from .fit import *
from .standard_functions import *

__all__ = ["fit", "functions", "plotly", "unbinned_maximum_likelihood_fit"]


def eval(x, result):
    fit_type, params, *_ = result
    return get_func(fit_type)(np.array(x), *params)
