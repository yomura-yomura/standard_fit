import numpy as np
import numpy_utility as npu
from . import regression
import iminuit
# import warnings


# dict([
#     ("is_valid", bool), ("has_valid_parameters", bool), ("has_reached_call_limit", bool),
#     ("is_above_max_edm", bool),
#     ("edm", "f8"), ("ncalls", "i4"),
# ])


def to_dtype(type: dict):
    return np.dtype(list(type.items()))


status_type = dict(
    is_valid=bool,
    has_valid_parameters=bool, has_reached_call_limit=bool, is_above_max_edm=bool,
    edm="f8",
    # ncalls='i4'
    nfcn="i4"
)

status_dtype = to_dtype(status_type)

# minos_type = dict(
#
# )

fit_type = dict(
    fit_type="U20",
    params="O",
    # cov_mat="O",
    err_params="O",
    fcn="f8",
    ndf="i8",
    x_range=("f8", 2),
    y_range=("f8", 2),
    status=status_dtype
)

fit_dtype = to_dtype(fit_type)


def to_numpy(obj):
    assert npu.is_array(obj)
    if isinstance(obj, np.ma.MaskedArray):
        if obj.mask.dtype.names is None:
            mask = obj.mask
        else:
            mask = npu.any_along_column(obj.mask)
        obj = obj.data
    else:
        obj = np.array(obj)
        mask = None

    base_fit_type = fit_type.copy()

    if mask is not None:
        obj = obj[~mask]

    original_shape = obj.shape

    if obj.dtype.names is not None:
        obj = np.array(obj.tolist())

    obj = obj.reshape((-1, len(base_fit_type)))

    fit_types = obj[..., 0]

    base_fit_type["fit_type"] = "U{}".format(max([len(ft) for ft in fit_types]))

    if len(np.unique(fit_types)) == 1:
        n_params = len(regression.get_parameter_names(fit_types[0]))
        base_fit_type["params"] = ("f8", (n_params,))
        base_fit_type["err_params"] = ("f8", (n_params,))

    obj = np.fromiter(map(tuple, obj), list(base_fit_type.items())).reshape(original_shape)
    if mask is None:
        return obj
    else:
        masked_obj = np.ma.empty(mask.shape, dtype=obj.dtype)
        masked_obj[~mask] = obj
        masked_obj.mask = mask
        return masked_obj


def _validate_data_set(a):
    if isinstance(a, np.ma.MaskedArray):
        a = a.compressed()
    elif isinstance(a, np.ndarray):
        pass
    else:
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


def fit(x, y, fit_type, error_x=None, error_y=None, parameter_error=None, fix_parameter=None,
        initial_guess=None, bounds=None, x_range=(), y_range=(), print_result=True, **kwargs):
    if len(x) != len(y):
        raise ValueError("mismatch length of x and y")

    fit_type = fit_type.replace(" ", "_")

    x = _validate_data_set(x)
    y = _validate_data_set(y)

    valid_selection = np.isfinite(x) & np.isfinite(y)

    if callable(x_range):
        x_range = x_range(x, y)
    if callable(y_range):
        y_range = y_range(x, y)

    x_range = _valid_range(x_range, x[valid_selection])
    y_range = _valid_range(y_range, y[valid_selection])

    range_selection = (
        ((x_range[0] <= x[valid_selection]) & (x[valid_selection] <= x_range[1])) &
        ((y_range[0] <= y[valid_selection]) & (y[valid_selection] <= y_range[1]))
    )

    selection = valid_selection
    selection[valid_selection] &= range_selection

    if np.count_nonzero(selection) == 0:
        return None

    x = x[selection]
    y = y[selection]
    if error_x is not None and not callable(error_y):
        if len(x) != len(error_x):
            raise ValueError("mismatch length of x and error_x")
        error_x = _validate_data_set(error_x)[selection]
    if error_y is not None and not callable(error_y):
        if len(y) != len(error_y):
            raise ValueError("mismatch length of y and error_y")
        error_y = _validate_data_set(error_y)[selection]

    if regression.linear.is_defined(fit_type):
        params, err_params, fval, ndf = regression.linear.get_fit_func(fit_type)(x, y, error_y=error_y, **kwargs)
        return (
            fit_type,
            params, err_params,
            fval, ndf,
            x_range, y_range,
            np.empty(1, np.dtype(list(status_type.items())))
        )

    if initial_guess is None:
        initial_guess = regression.nonlinear.estimate_initial_guess(fit_type, x, y)
        if fit_type == "gaussian":
            bounds = [None, None, (0, None)]

    fit_func = regression.nonlinear.get_func(fit_type)

    # regression.nonlinear.standard.functions.np = np
    # regression.nonlinear.custom.functions.np = np
    grad_fcn = None

    if error_x is None and error_y is None:
        def fcn(p):
            return np.sum((y - fit_func(x, *p)) ** 2)
    elif error_x is not None and error_y is not None:
        print("* Both error_x and error_y have been specified.")
        raise NotImplementedError
        # # import jax
        # # import jax.numpy as jnp
        # # # from .functions import gradient
        # # # differential_fit_func = gradient.get_func(fit_type)
        # #
        # # functions.standard.functions.np = jnp
        # # functions.custom.functions.np = jnp
        # #
        # # fit_func = jax.jit(fit_func)
        # # # differential_fit_func = jax.jit(jax.grad(fit_func))
        # #
        # # def make_differential_fit_func(n):
        # #     @jax.jit
        # #     def _inner(sigma_x, x, *params):
        # #         i_differential_fit_func = fit_func
        # #         result = 0.0
        # #         factorial = 1
        # #         for i in range(1, n+1):
        # #             factorial *= i
        # #             i_differential_fit_func = jax.jit(jax.grad(i_differential_fit_func))
        # #             result += i_differential_fit_func(x, *params) * sigma_x ** i / factorial
        # #         return result
        # #     return _inner
        # #
        # # tol = 1e-4
        # #
        # # def get_last_residual_term(params):
        # #     return np.array([
        # #         n_differential_fit_func(error_x, ix, *params) - n_minus_differential_fit_func(error_x, ix, *params)
        # #         for ix in x
        # #     ])
        # #
        # # for n in range(1, 5):
        # #     n_differential_fit_func = make_differential_fit_func(n)
        # #     n_minus_differential_fit_func = make_differential_fit_func(n-1)
        # #     x_propagated_sigma = np.abs(get_last_residual_term(initial_guess))
        # #     if x_propagated_sigma.max() < tol:
        # #         break
        # # else:
        # #     raise ValueError("n >= 5")
        # #
        # # print(f"* [Debug Message] n = {n}")
        # # print(f"* [Debug Message] x_propagated_sigma.max() = {x_propagated_sigma.max()}")
        # # # assert x_propagated_sigma.max() < 0.1
        # #
        # # @jax.jit
        # def fcn(par):
        #     # result = 0.0
        #     # for xi, yi, xei, yei in zip(x, y, error_x, error_y):
        #     #     y_var = yei ** 2 + (differential_fit_func(xi, *par) * xei) ** 2
        #     #     result += (yi - fit_func(xi, *par)) ** 2 / y_var
        #     # for xi, yi, xei, yei in zip(x, y, error_x, error_y):
        #     #     y_var = yei ** 2 + n_differential_fit_func(xei, xi, *par) ** 2
        #     #     result += (yi - fit_func(xi, *par)) ** 2 / y_var
        #
        #     y_var = error_y ** 2 + (fit_func(x + error_x, *par) - fit_func(x, *par)) ** 2
        #     result = ((y - fit_func(x, *par)) ** 2 / y_var).sum()
        #
        #     return result
        #
        # # grad_fcn = jax.jit(jax.grad(fcn))
        # # fcn = (fcn)
        # print('* Start optimizing using jax automatic differentiation (it may take a long time)')
    else:
        if error_x is not None:
            raise ValueError("error_x is specified, but error_y is not")
        else:
            if callable(error_y):
                def fcn(p):
                    return np.sum(((y - fit_func(x, *p)) / error_y(x, y, *p)) ** 2)
            else:
                def fcn(p):
                    return np.sum(((y - fit_func(x, *p)) / error_y) ** 2)

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

    m = iminuit.Minuit(
        fcn, initial_guess,
        grad=grad_fcn
    )
    if parameter_error is not None:
        m.errors = parameter_error
    if bounds is not None:
        m.limits = bounds
    if fix_parameter is not None:
        m.fixed = fix_parameter
    m.errordef = iminuit.Minuit.LEAST_SQUARES
    m.strategy = 2

    m.simplex()
    m.migrad()

    if print_result:
        print(m.fmin)
        print(m.params)

#     if "jax" in locals():
#         x_propagated_sigma = np.abs(get_last_residual_term(m.values.values()))
#         print(f"* [Debug Message] x_propagated_sigma.max() = {x_propagated_sigma.max()}")
#         if (x_propagated_sigma > tol).any():
#             warnings.warn(f"""
# the error of y that x propagated to is larger than expected ({tol:.1e}).
# ascending-order-sorted x_propagated_sigma:
# {np.sort(x_propagated_sigma)}
#     """)

    return (
        fit_type, tuple(m.values), tuple(m.errors), m.fval, len(x) - m.nfit, x_range, y_range,
        tuple(getattr(m.fmin, k) for k in status_type)
    )


def gaussian_fit(x, **kwargs):
    x = x[~np.isnan(x)]
    counts, bins = np.histogram(x, bins="auto")
    x = (bins[1:] + bins[:-1]) * 0.5
    y = counts

    return fit(x, y, "gaussian", **kwargs)

