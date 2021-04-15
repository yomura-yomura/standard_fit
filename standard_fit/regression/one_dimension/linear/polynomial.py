import numpy as np


def get_X(x, n_poly):
    return np.array([[ix ** j for j in range(n_poly + 1)] for ix in x])


def fit(x, y, n_poly, error_y=None):
    ndf = len(x) - (n_poly + 1)
    if ndf < 0:
        raise ValueError("ndf < 0")

    X = get_X(x, n_poly)
    if error_y is None:
        XX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XX_inv @ X.T @ y
        err_beta = np.sqrt(np.diag(XX_inv))
        rss = np.sum((y - X @ beta_hat) ** 2)
    else:
        W = np.diag(1 / error_y ** 2)
        XWX_inv = np.linalg.inv(X.T @ W @ X)
        beta_hat = XWX_inv @ X.T @ W @ y
        err_beta = np.sqrt(np.diag(XWX_inv))
        rss = np.sum((y - X @ beta_hat).T @ W @ (y - X @ beta_hat))

    return beta_hat, err_beta, rss, ndf


def eval(x, params):
    # return get_X(x, len(params) - 1) @ params
    return np.poly1d(list(reversed(params)))(x)
