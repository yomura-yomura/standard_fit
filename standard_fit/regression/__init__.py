from . import one_dimension, multi_dimension
import numpy as np


def eval(fit_type, x, params):
    x = np.asarray(x) if not isinstance(x, np.ma.MaskedArray) else x

    if not (1 <= x.ndim <= 2):
        raise ValueError("1 <= x.ndim <= 2")

    is_multivariate = x.ndim == 2

    if is_multivariate:
        return multi_dimension.eval(fit_type, x, params)
    else:
        return one_dimension.eval(fit_type, x, params)


def get_parameter_names(fit_type, is_multivariate):
    if is_multivariate:
        return multi_dimension.get_parameter_names(fit_type)
    else:
        return one_dimension.get_parameter_names(fit_type)
