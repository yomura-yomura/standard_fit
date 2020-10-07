from .standard_functions import *
from .custom_functions import *
import inspect

__all__ = []
__all__ += standard_functions.__all__
__all__ += custom_functions.__all__


def _validate_fit_type(fit_type):
    if fit_type not in __all__:
        raise ValueError(f"{fit_type} not defined in {__all__}")


def get_func(fit_type):
    fit_type = fit_type.replace(" ", "_")
    _validate_fit_type(fit_type)
    return globals()[fit_type]


def get_parameter_names(fit_type):
    fit_func = get_func(fit_type)
    _validate_fit_type(fit_type)
    return list(inspect.signature(fit_func).parameters)[1:]


def estimate_initial_guess(fit_type, x, y):
    _validate_fit_type(fit_type)
    try:
        return standard_functions.estimate_initial_guess(fit_type, x, y)
    except NotImplementedError:
        return custom_functions.estimate_initial_guess(fit_type, x, y)
