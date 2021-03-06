from . import polynomial, fourier
import re
import functools


__all__ = ["is_defined", "get_func", "eval", "get_parameter_names"]


def is_defined(fit_type):
    return bool(re.fullmatch(r"pol\d+", fit_type)) or bool(re.fullmatch(r"fourier\d+", fit_type))


def get_func(fit_type):
    if re.fullmatch(r"pol\d+", fit_type):
        return functools.partial(polynomial.fit, n_poly=int(fit_type[3:]))
    elif re.fullmatch(r"fourier\d+", fit_type):
        return functools.partial(fourier.fit, n_terms=int(fit_type[7:]))
    else:
        raise ValueError(f"{fit_type} not defined.")


def eval(fit_type, x, params):
    if re.fullmatch(r"pol\d+", fit_type):
        return polynomial.eval(x, params)
    elif re.fullmatch(r"fourier\d+", fit_type):
        return fourier.eval(x, params)
    else:
        raise ValueError(f"{fit_type} not defined.")


def get_parameter_names(fit_type):
    if re.match(r"pol\d+", fit_type):
        return [f"p{i}" for i in range(int(fit_type[3:]) + 1)]
    elif re.match(r"fourier\d+", fit_type):
        n_terms = int(fit_type[7:]) + 1
        return ["p0"] + [f"p_sin_{i}" for i in range(1, n_terms)] + [f"p_cos_{i}" for i in range(1, n_terms)]
    else:
        raise ValueError(f"{fit_type} not defined.")
