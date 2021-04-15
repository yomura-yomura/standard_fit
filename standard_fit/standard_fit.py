import numpy as np
import numpy_utility as npu
from . import regression
import iminuit
# import warnings
try:
    import IPython.display
    display = IPython.display.display
except ImportError:
    display = print


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

fit_result_type = dict(
    fit_type="U20",
    params="O",
    # cov_mat="O",
    err_params="O",
    fcn="f8",
    ndf="i8",
    # x_range=("f8", 2),
    x_range="O",
    y_range=("f8", 2),
    status=status_dtype,
    is_multivariate=bool
)

fit_result_dtype = to_dtype(fit_result_type)


def get_fit_result_dtype(params_info):
    fit_result_type_ = fit_result_type.copy()
    if npu.is_numeric(np.array(params_info)):
        fit_result_type_["params"] = fit_result_type_["err_params"] = ("f8", params_info)
    elif npu.is_array(params_info):
        fit_result_type_["params"] = fit_result_type_["err_params"] = [(param_name, "f8") for param_name in params_info]
    else:
        raise TypeError(type(params_info))
    return to_dtype(fit_result_type_)


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

    base_fit_result_type = fit_result_type.copy()

    if mask is not None:
        obj = obj[~mask]

    original_shape = obj.shape

    if obj.dtype.names is not None:
        obj = np.array(obj.tolist())

    obj = obj.reshape((-1, len(base_fit_result_type)))

    fit_types = obj[..., 0]

    base_fit_result_type["fit_type"] = "U{}".format(max([len(ft) for ft in fit_types]))

    if len(np.unique(fit_types)) == 1:
        n_params = len(regression.get_parameter_names(fit_types[0]))
        base_fit_result_type["params"] = ("f8", (n_params,))
        base_fit_result_type["err_params"] = ("f8", (n_params,))

    obj = np.fromiter(map(tuple, obj), list(base_fit_result_type.items())).reshape(original_shape)
    if mask is None:
        return obj
    else:
        masked_obj = np.ma.empty(mask.shape, dtype=obj.dtype)
        masked_obj[~mask] = obj
        masked_obj.mask = mask
        return masked_obj


def _validate_data_set(x, y, error_x, error_y):

    # Wrap x and y with np.ndarray or np.ma.MaskedArray and error_x and error_y also if possible
    # Check ndim and shape

    x = np.asarray(x) if not isinstance(x, np.ma.MaskedArray) else x
    y = np.asarray(y) if not isinstance(y, np.ma.MaskedArray) else y

    if x.ndim == 1:
        is_multivariate = False
    elif x.ndim == 2:
        is_multivariate = True
    else:
        raise ValueError("1 <= x.ndim <= 2")

    if y.ndim == 1:
        pass
    elif y.ndim == 2 and y.shape[1] == 1:
        y = y.flatten()
    else:
        raise ValueError("y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)")
    
    if x.shape[0] != y.shape[0]:
        raise ValueError("mismatch length of x and y")
    
    if npu.is_array(error_x):
        error_x = np.asarray(error_x) if not isinstance(error_x, np.ma.MaskedArray) else error_x

        if not (1 <= error_x.ndim <= 2):
            raise ValueError("1 <= error_x.ndim <= 2")

        if x.shape != error_x.shape:
            raise ValueError(f"mismatch shape of x and error_x: {x.shape} and {error_y.shape}")
        
    if npu.is_array(error_y):
        error_y = np.asarray(error_y) if not isinstance(error_y, np.ma.MaskedArray) else error_y

        if error_y.ndim == 1:
            pass
        elif error_y.ndim == 2 and error_y.shape[1] == 1:
            error_y = error_y.flatten()
        else:
            raise ValueError("error_y.ndim == 1 or (error_y.ndim == 2 and error_y.shape[1] == 1)")

        if y.shape[0] != error_y.shape[0]:
            raise ValueError(f"mismatch length of y and error_y: {y.shape[0]} and {error_y.shape[0]}")

    # Check type of array

    if isinstance(x, np.ma.MaskedArray) and isinstance(y, np.ma.MaskedArray):
        x_mask = x.mask.any(axis=-1) if is_multivariate else x.mask
        common_mask = x_mask | y.mask
        
        if error_x is None or callable(error_x):
            pass
        elif isinstance(error_x, np.ma.MaskedArray):
            common_mask |= error_x.mask.any(axis=-1) if is_multivariate else error_x.mask
        else:
            raise ValueError("error_x should be np.ma.MaskedArray or None")
        
        if error_y is None or callable(error_y):
            pass
        elif isinstance(error_y, np.ma.MaskedArray):
            common_mask |= error_y.mask
        else:
            raise ValueError("error_y should be np.ma.MaskedArray or None")
        
        x = x.data[~common_mask]
        y = y.data[~common_mask]
        
        if isinstance(error_x, np.ma.MaskedArray):
            error_x = error_x.data[~common_mask]
        if isinstance(error_y, np.ma.MaskedArray):
            error_y = error_y.data[~common_mask]
    elif not isinstance(x, np.ma.MaskedArray) and not isinstance(y, np.ma.MaskedArray):
        if not (not isinstance(error_x, np.ma.MaskedArray) or error_x is None or callable(error_x)):
            raise ValueError("error_x should not be np.ma.MaskedArray or should be None or callable object")
        if not (not isinstance(error_y, np.ma.MaskedArray) or error_y is None or callable(error_y)):
            raise ValueError("error_y should not be np.ma.MaskedArray or should be None or callable object")
    else:
        raise ValueError("both x and y must be np.ma.MaskedArray instances or other array-like objects")
    
    # Check invalid values

    x_isfinite = np.isfinite(x).all(axis=-1) if is_multivariate else np.isfinite(x)
    common_isfinite = np.isfinite(y) & x_isfinite
    
    x = x[common_isfinite]
    y = y[common_isfinite]
    
    if isinstance(error_x, np.ndarray):
        error_x = error_x[common_isfinite]
        if not np.isfinite(error_x).all():
            raise NotImplementedError("it's not supported yet that there are some invalid values in error_x")

    if isinstance(error_y, np.ndarray):
        error_y = error_y[common_isfinite]
        if not np.isfinite(error_y).all():
            raise NotImplementedError("it's not supported yet that there are some invalid values in error_y")

    #     if np.issubdtype(a.dtype, np.object_):
    #         try:
    #             a = a.astype("M8")
    #         except TypeError:
    #             raise ValueError(f"""
    # Invalid data set is passed:
    # {a}
    #             """)
    return x, y, error_x, error_y, is_multivariate


def _validate_range(x_range, x: np.ndarray):
    ndim = x.ndim
    if ndim == 1:
        assert np.ndim(x_range) <= 1
    elif ndim == 2:
        if x_range is None:
            x_range = [None] * x.shape[-1]
        assert len(x_range) == np.shape(x)[-1]

        return [_validate_range(ixr, ix) for ixr, ix in zip(x_range, np.rollaxis(x, axis=-1))]

    if x_range is None or len(x_range) == 0:
        x_range = (-np.inf, np.inf)
    elif np.shape(x_range)[-1] == 2:
        x_range = list(x_range)
        if x_range[0] is None:
            x_range[0] = np.min(x)
        if x_range[1] is None:
            x_range[1] = np.max(x)
        x_range = tuple(x_range)
    else:
        raise ValueError("x/y-range must be like (float, float) or [(float, float), ...]")
    return x_range


def fit(x, y, fit_type, error_x=None, error_y=None, parameter_error=None, fix_parameter=None,
        initial_guess=None, bounds=None, x_range=None, y_range=None,
        print_result=True, print_level=None, **kwargs):
    if len(x) != len(y):
        raise ValueError("mismatch length of x and y")

    fit_type = fit_type.replace(" ", "_")

    x, y, error_x, error_y, is_multivariate = _validate_data_set(x, y, error_x, error_y)

    if callable(x_range):
        x_range = x_range(x, y)
    if callable(y_range):
        y_range = y_range(x, y)

    x_range = _validate_range(x_range, x)
    y_range = _validate_range(y_range, y)

    # selection for range

    if is_multivariate:
        xf, xl = (np.expand_dims(e, axis=0) for e in zip(*x_range))
        range_selection = np.all((xf <= x) & (x <= xl), axis=-1)
    else:
        range_selection = (x_range[0] <= x) & (x <= x_range[1])

    range_selection &= (y_range[0] <= y) & (y <= y_range[1])

    x = x[range_selection]
    y = y[range_selection]
    if isinstance(error_x, np.ndarray):
        error_x = error_x[range_selection]
    if isinstance(error_y, np.ndarray):
        error_y = error_y[range_selection]

    if x.size == 0:
        return None

    if is_multivariate:
        if regression.multi_dimension.linear.is_defined(fit_type):
            raise NotImplementedError
        elif regression.multi_dimension.nonlinear.is_defined(fit_type):
            if initial_guess is None:
                initial_guess = regression.multi_dimension.nonlinear.estimate_initial_guess(fit_type, x, y)

            fit_func = regression.multi_dimension.nonlinear.get_func(fit_type)
    else:
        if regression.one_dimension.linear.is_defined(fit_type):
            params, err_params, fval, ndf = regression.one_dimension.linear.get_func(fit_type)(x, y, error_y=error_y, **kwargs)
            if print_result:
                print(f"fval/ndf = {fval:.4g}/{ndf}")
                print(f"params = {params}")
                print(f"err_params = {err_params}")
            return (
                fit_type,
                params, err_params,
                fval, ndf,
                x_range, y_range,
                np.empty(1, np.dtype(list(status_type.items()))),
                is_multivariate
            )
        elif regression.one_dimension.nonlinear.is_defined(fit_type):
            if initial_guess is None:
                initial_guess = regression.one_dimension.nonlinear.estimate_initial_guess(fit_type, x, y)

            if fit_type == "gaussian":
                bounds = [None, None, (0, None)]

            fit_func = regression.one_dimension.nonlinear.get_func(fit_type)
        else:
            raise ValueError(f"{fit_type} not defined")

    if error_x is None and error_y is None:
        def fcn(p):
            return np.sum((y - fit_func(x, *p)) ** 2)
    elif error_x is not None and error_y is not None:
        print("* Both error_x and error_y have been specified.")
        raise NotImplementedError
    else:
        if error_x is not None:
            raise ValueError("error_x is specified, but error_y is not")
        elif error_y is not None:
            if callable(error_y):
                def fcn(p):
                    return np.sum(((y - fit_func(x, *p)) / error_y(x, y, *p)) ** 2)
            else:
                def fcn(p):
                    return np.sum(((y - fit_func(x, *p)) / error_y) ** 2)
        else:
            raise NotImplementedError

    def migrad(use_simplex=True, strategy=2):
        m = iminuit.Minuit(fcn, initial_guess)

        if print_level is not None:
            assert 0 <= print_level <= 3
            m.print_level = print_level

        m.errordef = iminuit.Minuit.LEAST_SQUARES
        m.strategy = strategy

        if parameter_error is not None:
            m.errors = parameter_error
        if bounds is not None:
            m.limits = bounds
        if fix_parameter is not None:
            m.fixed = fix_parameter

        if use_simplex:
            m.simplex().migrad()
        else:
            m.migrad()

        if m.fmin.hesse_failed:
            m.errors[:] = [np.nan] * m.nfit

        return m

    m = migrad()

    if print_result:
        display(m.fmin)
        display(m.params)

    return (
        fit_type, tuple(m.values), tuple(m.errors), m.fval, len(x) - m.nfit, x_range, y_range,
        tuple(getattr(m.fmin, k) for k in status_type), is_multivariate
    )


def gaussian_fit(x, **kwargs):
    x = x[~np.isnan(x)]
    counts, bins = np.histogram(x, bins="auto")
    x = (bins[1:] + bins[:-1]) * 0.5
    y = counts

    return fit(x, y, "gaussian", **kwargs)

