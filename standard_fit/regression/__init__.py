from . import linear, nonlinear


def eval(fit_type, x, params):
    if linear.is_defined(fit_type):
        return linear.eval(fit_type, x, params)
    elif nonlinear.is_defined(fit_type):
        return nonlinear.get_func(fit_type)(x, *params)
    else:
        raise ValueError(f"{fit_type} not defined.")


def get_parameter_names(fit_type):
    if linear.is_defined(fit_type):
        return linear.get_parameter_names(fit_type)
    elif nonlinear.is_defined(fit_type):
        return nonlinear.get_parameter_names(fit_type)
    else:
        raise ValueError(f"{fit_type} not defined.")
