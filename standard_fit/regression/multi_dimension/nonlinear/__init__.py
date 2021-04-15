from . import standard, custom
import inspect

__all__ = ["is_defined", "get_func", "eval", "get_parameter_names", "estimate_initial_guess"]


def is_defined(fit_type):
    return (fit_type in standard.functions.__all__) or (fit_type in custom.functions.__all__)


def _validate_fit_type(fit_type):
    if fit_type not in standard.functions.__all__ and fit_type not in custom.__all__:
        raise ValueError(f"{fit_type} not defined in {standard.functions.__all__ + custom.__all__}")


def get_func(fit_type):
    fit_type = fit_type.replace(" ", "_")
    _validate_fit_type(fit_type)
    if fit_type in standard.functions.__all__:
        return standard.functions.get_func(fit_type)
    elif fit_type in custom.__all__:
        return custom.functions.get_func(fit_type)
    else:
        raise ValueError(f"{fit_type} not defined.")


def eval(fit_type, x, params):
    return get_func(fit_type)(x, *params)


def get_parameter_names(fit_type):
    fit_func = get_func(fit_type)
    _validate_fit_type(fit_type)
    return list(inspect.signature(fit_func).parameters)[1:]


def estimate_initial_guess(fit_type, x, y):
    _validate_fit_type(fit_type)
    if fit_type in standard.functions.__all__:
        return standard.estimate_initial_guess(fit_type, x, y)
    elif fit_type in custom.__all__:
        return custom.estimate_initial_guess(fit_type, x, y)
