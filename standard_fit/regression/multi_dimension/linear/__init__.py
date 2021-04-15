

__all__ = ["is_defined", "get_fit_func", "eval", "get_parameter_names"]


def is_defined(fit_type):
    return False


def get_fit_func(fit_type):
    raise ValueError(f"{fit_type} not defined.")


def eval(fit_type, x, params):
    raise ValueError(f"{fit_type} not defined.")


def get_parameter_names(fit_type):
    raise ValueError(f"{fit_type} not defined.")
