import numpy as np
import scipy.stats
import parse


_gaus_format = "gaussian{:d}d"
_gaus_parser = parse.compile(_gaus_format)

__all__ = [_gaus_format.format(i) for i in range(1, 4)]


def __getattr__(name):
    matched = _gaus_parser.parse(name)
    if matched:
        func = _make_n_dimensional_gaussian(int(matched[0]))
        if name not in __all__:
            __all__.append(name)
        return func

    raise AttributeError(f"it has no attribute '{name}'")


def make_symmetry_from(tri, n):
    tri = np.asarray(tri)
    a = np.empty((n, n), dtype=tri.dtype)
    a[np.triu_indices_from(a)] = a[np.tril_indices_from(a)] = tri
    return a


def _make_n_dimensional_gaussian(n, force=False):
    assert 0 < n
    mean_args = ", ".join(f"μ_{i}" for i in range(n))
    cov_args = ", ".join(f"Σ_{r}_{c}" for r in range(n) for c in range(n) if c <= r)
    func_name = _gaus_format.format(n)
    if force is True or func_name not in globals():
        exec(f"""
def {func_name}(x, {mean_args}, {cov_args}):
    return generic_gaussian(x, [{mean_args}], make_symmetry_from([{cov_args}], {n}))
        """, globals())
    return globals()[func_name]


def generic_gaussian(x, μ, Σ):
    return scipy.stats.multivariate_normal.pdf(x, μ, Σ)


def get_func(fit_type: str):
    assert fit_type in __all__
    return globals()[fit_type]
