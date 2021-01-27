# Maybe will be removed
from .fit import *
from . import regression
import inspect


__all__ = ["fit", "regression", "plotly", "unbinned_maximum_likelihood_fit"]

__all__ += ["eval"]


def eval(x, result):
    fit_type, params, *_ = result
    return regression.eval(fit_type, np.array(x), *params)


def add_function(fit_type: str, func, initial_guess=None, bounds=None, overwrite=False):
    assert bounds is None

    fit_type = fit_type.replace(" ", "_")
    if overwrite is False and regression.nonlinear.is_defined(fit_type):
        raise ValueError(f"'{fit_type}' has already been defined")

    if not callable(func):
        raise ValueError("'func' argument must be callable")

    n_params = len(inspect.signature(func).parameters) - 1
    if n_params <= 0:
        raise ValueError("'func' argument must be a function with more than 1 parameter")

    if initial_guess is None:
        initial_guess_func = lambda x, y: (0,) * n_params
    else:
        if callable(initial_guess):
            if len(inspect.signature(initial_guess).parameters) != 2:
                raise ValueError("'func' argument must be a function with 2 parameters")
            initial_guess_func = initial_guess
        elif npu.is_array(initial_guess):
            if len(initial_guess) != n_params:
                raise ValueError(f"'initial_guess' must be array-like with more than {n_params}")
            initial_guess_func = lambda x, y: initial_guess
        else:
            raise NotImplementedError("")

    setattr(regression.nonlinear.custom.initial_guess, fit_type, initial_guess_func)
    setattr(regression.nonlinear.custom.functions, fit_type, func)
    regression.nonlinear.custom.__all__.append(fit_type)

