import numpy as np
import numpy_utility as npu
from . import functions
import iminuit


fit_dtype = dict(
    fit_type="U20",
    params="O",
    # cov_mat="O",
    err_params="O",
    fcn="f8",
    ndf="i8",
    x_range=("f8", 2),
    y_range=("f8", 2)
)


def any_along_column(a):
    assert a.dtype.names is not None
    assert len(a.dtype.names) > 0

    ndim = a.ndim
    return np.any(
        [a[n] if a[n].ndim <= ndim else a[n].any(axis=tuple(np.arange(ndim, a[n].ndim)))
         for n in a.dtype.names],
        axis=0
    )


def to_numpy(obj):
    assert npu.is_array(obj)
    if isinstance(obj, np.ma.MaskedArray):
        if obj.mask.dtype.names is None:
            mask = obj.mask
        else:
            mask = any_along_column(obj.mask)
        obj = obj.data
    else:
        obj = np.array(obj)
        mask = None

    base_dtype = fit_dtype.copy()

    if mask is not None:
        obj = obj[~mask]

    original_shape = obj.shape

    if obj.dtype.names is not None:
        obj = np.array(obj.tolist())

    obj = obj.reshape((-1, len(base_dtype)))

    fit_types = obj[..., 0]

    base_dtype["fit_type"] = "U{}".format(max([len(ft) for ft in fit_types]))

    if len(np.unique(fit_types)) == 1:
        # base_dtype["params"] = [(n, "f8") for n in functions.get_parameter_names(fit_types[0])]
        # base_dtype["cov_mat"] = [(n, base_dtype["params"]) for n in functions.get_parameter_names(fit_types[0])]
        n_params = len(functions.get_parameter_names(fit_types[0]))
        base_dtype["params"] = ("f8", n_params)
        # base_dtype["cov_mat"] = ("f8", (n_params, n_params))
        base_dtype["err_params"] = ("f8", n_params)

    obj = np.fromiter(map(tuple, obj), list(base_dtype.items())).reshape(original_shape)
    if mask is None:
        return obj
    else:
        masked_obj = np.ma.empty(mask.shape, dtype=obj.dtype)
        masked_obj[~mask] = obj
        masked_obj.mask = mask
        return masked_obj


def _validate_data_set(a):
    a = np.array(a)
    if np.issubdtype(a.dtype, np.object_):
        try:
            a = a.astype("M8")
        except TypeError:
            raise ValueError(f"""
Invalid data set is passed:
{a}
            """)
    return a


def _valid_range(x_range, x):
    if len(x_range) == 0:
        # x_range = (np.min(x), np.max(x))
        x_range = (-np.inf, np.inf)
    elif len(x_range) == 2:
        x_range = list(x_range)
        if x_range[0] is None:
            x_range[0] = np.min(x)
        if x_range[1] is None:
            x_range[1] = np.max(x)
    else:
        raise ValueError("x/y-range must be like (float, float)")
    return x_range


def fit(x, y, fit_type, x_err=None, y_err=None,
        initial_guess=None, bounds=None, x_range=(), y_range=(), print_result=True):
    assert len(x) == len(y)

    fit_type = fit_type.replace(" ", "_")
    if fit_type not in functions.__all__:
        raise ValueError(f"Not available fit type: {fit_type}")

    x = _validate_data_set(x)
    y = _validate_data_set(y)

    # if np.issubdtype(x.dtype, np.datetime64) or np.issubdtype(y.dtype, np.datetime64):
    #     return fit_time_series(x, y, fit_type, y_err, initial_guess, bounds, x_range, y_range, print_result)

    valid_selection = np.isfinite(x) | np.isfinite(y)

    x_range = _valid_range(x_range, x[valid_selection])
    y_range = _valid_range(y_range, y[valid_selection])

    range_selection = (
        ((x_range[0] <= x) & (x <= x_range[1])) &
        ((y_range[0] <= y) & (y <= y_range[1]))
    )

    selection = valid_selection & range_selection
    x = x[selection]
    y = y[selection]

    if np.count_nonzero(selection) == 0:
        return None

    if initial_guess is None:
        initial_guess = functions.estimate_initial_guess(fit_type, x, y)
        if fit_type == "gaussian":
            bounds = [None, None, (0, None)]

    fit_func = functions.get_func(fit_type)

    if x_err is None and y_err is None:
        def fcn(p):
            return np.sum((fit_func(x, *p) - y) ** 2)
    elif x_err is not None and y_err is not None:
        assert len(selection) == len(x_err) == len(y_err)
        import jax
        from .functions import gradient
        differential_fit_func = gradient.get_func(fit_type)

        @jax.jit
        def fcn(par):
            result = 0.0
            for xi, yi, xei, yei in zip(x, y, x_err, y_err):
                y_var = yei ** 2 + (differential_fit_func(xi, *par) * xei) ** 2
                result += (yi - fit_func(xi, *par)) ** 2 / y_var
            return result
    else:
        if x_err is not None:
            assert len(selection) == len(x_err)
            x_err = x_err[selection]

            def fcn(p):
                return np.sum(((fit_func(x, *p) - y) / x_err) ** 2)
        else:
            assert len(selection) == len(y_err)
            y_err = y_err[selection]

            def fcn(p):
                return np.sum(((fit_func(x, *p) - y) / y_err) ** 2)

    # from .functions import gradient
    # ff = gradient.get_func(fit_type)
    # gff = jax.jit(jax.grad(ff))

    # @jax.jit
    # @jax.grad
    # def grad_fcn(p):
    #     return jnp.sum((ff(x, *p) - y) ** 2)
    # @jax.jit
    # def grad_fcn(p):
    #     return jnp.sum(-2 * jnp.array([gff(x_, *p) for x_ in x]) * (y - ff(x, *p)))

    m = iminuit.Minuit.from_array_func(
        fcn,
        # grad=grad_fcn,
        start=initial_guess,
        limit=bounds,
        errordef=iminuit.Minuit.LEAST_SQUARES
    )
    m.strategy = 2
    m.migrad()
    m.hesse()

    if print_result:
        m.print_fmin()
        m.print_param()

    return fit_type, tuple(m.values.values()), tuple(m.errors.values()), m.fval, len(x) - m.nfit, x_range, y_range


# def fit_time_series(x, y, fit_type, y_err=None, initial_guess=None, bounds=None, x_range=(), y_range=(), print_result=True):
#     x = _validate_data_set(x)
#     y = _validate_data_set(y)
#
#     if np.issubdtype(x.dtype, np.datetime64):
#         # x_ = (x - np.min(x)).astype(int)
#         x_ = x.astype(int)
#         print("x is scaled on {}, starts from {}".format(
#             "".join(str(e) for e in reversed(np.datetime_data(x.dtype))),
#             np.min(x)
#         ))
#     else:
#         x_ = x
#
#     if np.issubdtype(y.dtype, np.datetime64):
#         # y_ = (y - np.min(y)).astype(int)
#         y_ = y.astype(int)
#         print("y is scaled on {}, starts from {}".format(
#             "".join(str(e) for e in reversed(np.datetime_data(y.dtype))),
#             np.min(y)
#         ))
#     else:
#         y_ = y
#
#     assert y_err is None
#     return fit(x_, y_, fit_type, y_err, initial_guess, bounds, x_range, y_range, print_result)




def gaussian_fit(x, **kwargs):
    x = x[~np.isnan(x)]
    counts, bins = np.histogram(x, bins="auto")
    x = (bins[1:] + bins[:-1]) * 0.5
    y = counts

    return fit(x, y, "gaussian", **kwargs)


# def gaussian_fit_and_fig(x, px_kwargs={}, **kwargs):
#     from standard_fit.plotly.express import _plotly_express as _px
#     fig = _px.histogram(x=x, fit_type="gaussian", fit_stats=True, **px_kwargs)
#     return fig
#
#
# def gaussian_fit_and_show(x, **kwargs):
#     fig = gaussian_fit_and_fig(x, **kwargs)
#     fig.show(config=dict(editable=True))
#
#
# def fit_and_fig(x, y, fit_type, px_kwargs={}, *args, **kwargs):
#     from standard_fit.plotly.express import _plotly_express as _px
#     result = fit(x, y, fit_type, *args, **kwargs)
#     fig = _px.scatter(x, y, result, **px_kwargs)
#     return fig
#
#
# def fit_and_show(x, y, fit_type, px_kwargs={}, *args, **kwargs):
#     from standard_fit.plotly.express import _plotly_express as _px
#     result = fit(x, y, fit_type, *args, **kwargs)
#     fig = _px.scatter(x, y, result, **px_kwargs)
#     fig.show(config=dict(editable=True))

    




