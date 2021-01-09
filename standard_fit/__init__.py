# Maybe will be removed
from .fit import *
from . import regression
import inspect


__all__ = ["fit", "regression", "plotly", "unbinned_maximum_likelihood_fit"]

__all__ += ["eval"]


def eval(x, result):
    fit_type, params, *_ = result
    return regression.eval(fit_type, np.array(x), *params)


def add_function(fit_type: str, func, initial_guess_func=None):
    fit_type = fit_type.replace(" ", "_")
    if regression.nonlinear.is_defined(fit_type):
        raise ValueError(f"'{fit_type}' has already been defined")

    if not callable(func):
        raise ValueError("'func' argument must be callable")

    n_params = len(inspect.signature(func).parameters) - 1
    if n_params <= 0:
        raise ValueError("'func' argument must be a function with more than 1 parameter")

    if initial_guess_func is None:
        initial_guess_func = lambda x, y: (0,) * n_params
    else:
        if not callable(initial_guess_func):
            raise ValueError("'initial_guess_func' argument must be callable")
        if len(inspect.signature(initial_guess_func).parameters) != 2:
            raise ValueError("'func' argument must be a function with 2 parameters")

    setattr(regression.nonlinear.custom.initial_guess, fit_type, initial_guess_func)
    setattr(regression.nonlinear.custom.functions, fit_type, func)
    regression.nonlinear.custom.__all__.append(fit_type)

