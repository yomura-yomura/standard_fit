# Estimate initial guess
import numpy as np
from . import functions


__all__ = ["estimate_initial_guess"]


def gaussian(x, y):
    x = np.asarray(x)
    assert x.ndim == 2
    mean = np.average(x, weights=y, axis=0)
    # diag_cov = np.average((x - mean[np.newaxis, :]) ** 2, weights=y, axis=0)
    # return *mean, *np.diag(diag_cov).flatten()
    cov = np.average(
        (x - mean[np.newaxis, :])[..., np.newaxis] *
        (x - mean[np.newaxis, :])[:, np.newaxis],
        weights=y, axis=0
    )
    return *mean, *cov[np.triu_indices_from(cov)]


def estimate_initial_guess(fit_type: str, x, y):
    if functions._gaus_parser.parse(fit_type):
        return gaussian(x, y)
    elif fit_type in (k for k in globals() if not k.startswith("_")):
        return globals()[fit_type](x, y)
    else:
        raise NotImplementedError(fit_type)
