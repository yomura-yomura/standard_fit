from .standard_functions import *
from .custom_functions import *
from . import standard_functions as sf
from . import custom_functions as cf

fit_list = sf.fit_list + cf.fit_list


def get_func(fit_type):
    try:
        return sf.get_func(fit_type)
    except KeyError:
        return cf.get_func(fit_type)


def get_parameter_names(fit_type):
    try:
        return sf.get_parameter_names(fit_type)
    except KeyError:
        return cf.get_parameter_names(fit_type)


def estimate_initial_guess(fit_type, x, y):
    try:
        return sf.estimate_initial_guess(fit_type, x, y)
    except NotImplementedError:
        return cf.estimate_initial_guess(fit_type, x, y)
