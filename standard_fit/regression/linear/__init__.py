from . import polynomial, fourier
import re
import functools


def is_defined(fit_type):
    return (bool(re.match(r"pol\d+", fit_type)) or bool(re.match(r"fourier\d+", fit_type)))


def get_fit_func(fit_type):
    if re.match(r"pol\d+", fit_type):
        return functools.partial(polynomial.fit, n_poly=int(fit_type[3:]))
    elif re.match(r"fourier\d+", fit_type):
        return functools.partial(fourier.fit, n_terms=int(fit_type[7:]))
    else:
        raise ValueError(f"{fit_type} not defined.")


def eval(fit_type, x, params):
    if re.match(r"pol\d+", fit_type):
        return polynomial.eval(x, params)
    elif re.match(r"fourier\d+", fit_type):
        return fourier.eval(x, params)
    else:
        raise ValueError(f"{fit_type} not defined.")


def get_parameter_names(fit_type):
    if re.match(r"pol\d+", fit_type):
        return [f"p{i}" for i in range(int(fit_type[3:]) + 1)]
    elif re.match(r"fourier\d+", fit_type):
        n_terms = int(fit_type[7:]) + 1
        return ["p0"] + [f"p_sin{i}" for i in range(1, n_terms)] + [f"p_cos{i}" for i in range(1, n_terms)]
    else:
        raise ValueError(f"{fit_type} not defined.")
