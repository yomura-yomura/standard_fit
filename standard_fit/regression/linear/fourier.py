import numpy as np


def get_X(x, n_terms):
    return np.array([
        [1] +
        [np.sin(j * ix) for j in range(1, n_terms + 1)] +
        [np.cos(j * ix) for j in range(1, n_terms + 1)]
        for ix in x]
    )


def fit(x, y, n_terms, error_y=None, ridge_lambda=None, lasso_lambda=None):
    X = get_X(x, n_terms)
    n, p = X.shape

    n_fit = 2 * n_terms + 1
    ndf = n - n_fit
    # if ndf < 0:
    #     raise ValueError("ndf < 0")

    if error_y is None:
        XX_inv = np.linalg.inv(X.T @ X)
        if np.linalg.matrix_rank(XX_inv) < n_fit:
            raise ValueError(f"Matrix rank is smaller than n_fit: {np.linalg.matrix_rank(XX_inv)} < {n_fit}")
        beta_hat = XX_inv @ X.T @ y
        err_beta = np.sqrt(np.diag(XX_inv))
        rss = np.sum((y - X @ beta_hat) ** 2)
    else:
        W = np.diag(1 / error_y ** 2)
        if ridge_lambda is None and lasso_lambda is None:
            XWX_inv = np.linalg.inv(X.T @ W @ X)
            if np.linalg.matrix_rank(XWX_inv) < n_fit:
                raise ValueError(f"Matrix rank is smaller than n_fit: {np.linalg.matrix_rank(XWX_inv)} < {n_fit}")
            beta_hat = XWX_inv @ X.T @ W @ y
            # err_beta = np.sqrt(np.diag(XWX_inv))
            cov = XWX_inv @ X.T @ W @ X @ XWX_inv
            err_beta = np.sqrt(np.diag(cov))
        elif ridge_lambda is not None:
            XWX_inv = np.linalg.inv(X.T @ W @ X + n * ridge_lambda * np.identity(p))
            beta_hat = XWX_inv @ X.T @ W @ y
            err_beta = np.array([np.nan] * p)
        elif lasso_lambda is not None:
            from sklearn import linear_model
            if lasso_lambda == "auto":
                reg = linear_model.LassoCV()
            else:
                assert lasso_lambda > 0
                reg = linear_model.Lasso(alpha=lasso_lambda)
            reg.fit(X[:, 1:], y)
            beta_hat = np.append(reg.intercept_, reg.coef_)
            # A = X.T @ W @ y
            # XWX_inv = (beta_hat * A[:, np.newaxis]) @ np.linalg.inv(A * A[:, np.newaxis])
            err_beta = np.array([np.nan] * p)
        else:
            raise NotImplementedError

        rss = np.sum((y - X @ beta_hat).T @ W @ (y - X @ beta_hat))

    return beta_hat, err_beta, rss, ndf


def eval(x, params):
    return get_X(x, (len(params) + 1) // 2 - 1) @ params
