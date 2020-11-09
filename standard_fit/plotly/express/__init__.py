from standard_fit.plotly import util
import plotly_utility.express as pux
import standard_fit.fit
import plotly_utility
import numpy as np
import itertools
import numpy_utility as npu


__all__ = ["fit", "scatter", "histogram"]


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


def fit(fig, fit_type, row=1, col=1, i_data=1, trace_type=None, fit_stats=True, add_trace=True,
        datetime_type=None,
        fit_kwargs={}, annotation_kwargs={},
        fit_plot_kwargs={}):

    if trace_type is None:
        pass
    elif not npu.is_array(trace_type):
        trace_type = [trace_type]

    if row == "all" or col == "all" or i_data == "all":
        n_row, n_col, n_data, *_ = np.shape(fig._grid_ref)
        row = [row] if row != "all" else list(range(1, n_row+1))
        col = [col] if col != "all" else list(range(1, n_col+1))
        i_data = [i_data] if i_data != "all" else list(range(1, n_data+1))
        for r, c, d in itertools.product(row, col, i_data):
            fit(fig, fit_type, r, c, d, trace_type, fit_stats,  add_trace,
                datetime_type,
                fit_kwargs, annotation_kwargs)
        return fig

    traces = plotly_utility.get_traces_at(fig, row, col)

    if len(traces) == 0:
        # raise ValueError
        return fig

    trace = traces[i_data - 1]

    if trace_type is None:
        pass
    elif trace.type not in trace_type:
        return fig

    if trace.type not in ("scatter", "scattergl", "bar"):
        raise NotImplementedError(f"{type(trace)} not supported.")

    log_x = fig.layout.xaxis.type == "log"
    flip_xy = trace.type == "bar" and trace.orientation == "h"

    x = _validate_data_set(trace.x)
    y = _validate_data_set(trace.y)

    x_err = trace.error_x.array
    y_err = trace.error_y.array

    if flip_xy:
        x, y = y, x
        x_err, y_err = y_err, x_err

    if np.issubdtype(x.dtype, np.datetime64) or np.issubdtype(y.dtype, np.datetime64):
        if np.issubdtype(x.dtype, np.datetime64):
            if datetime_type is not None:
                x = x.astype(f"M8[{datetime_type}]")
            x_ = (x - x.min()).astype(int)
        else:
            x_ = x

        if np.issubdtype(y.dtype, np.datetime64):
            if datetime_type is not None:
                y = y.astype(f"M8[{datetime_type}]")
            y_ = (y - y.min()).astype(int)
            assert y_err is None
        else:
            y_ = y

        result = standard_fit.fit(x_, y_, fit_type, x_err=x_err, y_err=y_err, **fit_kwargs)
        if result is None:
            return fig

        if add_trace:
            fit_trace = util.get_fit_trace(result, x_, log_x=log_x, flip_xy=flip_xy, showlegend=False, **fit_plot_kwargs)
            fit_trace.x = fit_trace.x.astype(int) + x.min()
            fig.add_trace(fit_trace, row, col)
        if fit_stats:
            util.add_annotation(fig, result, row, col, **annotation_kwargs)
            time_unit = "".join(str(e) for e in reversed(np.datetime_data(x.dtype)))
            fig.layout.annotations[-1].text += "".join([
                "<br>",
                f"(starts from {x.min()}", "<br>",
                f" with time unit {time_unit})"
            ])
    else:
        result = standard_fit.fit(x, y, fit_type, x_err=x_err, y_err=y_err, **fit_kwargs)
        if result is None:
            return fig

        if add_trace:
            fig.add_trace(
                util.get_fit_trace(result, x, log_x=log_x, flip_xy=flip_xy, showlegend=False, **fit_plot_kwargs),
                row, col
            )

        if fit_stats:
            util.add_annotation(fig, result, row, col, **annotation_kwargs)

    if row is None and col is None:
        if not hasattr(fig, "_fit_results"):
            n_data = len(fig.data) - 1
            fig._fit_results = np.ma.zeros(n_data, dtype=list(standard_fit.fit_dtype.items()))
            fig._fit_results.mask = True
        fig._fit_results[i_data-1] = result
    else:
        if not hasattr(fig, "_fit_results"):
            n_row, n_col, n_data, *_ = np.shape(fig._grid_ref)
            fig._fit_results = np.ma.zeros((n_row, n_col, n_data), dtype=list(standard_fit.fit_dtype.items()))
            fig._fit_results.mask = True
        fig._fit_results[row-1][col-1][i_data-1] = result

    return fig


def scatter(x, y, fit_type=None, fit_stats=True, fit_kwargs={},
            fit_marginal_x=None, fit_marginal_y=None,
            **kwargs):
    if fit_marginal_x is not None:
        kwargs["marginal_x"] = "histogram"
    if fit_marginal_y is not None:
        kwargs["marginal_y"] = "histogram"

    fig = pux.scatter(x=x, y=y, **kwargs)

    if fit_type is not None:
        fit(fig, fit_type, 1, 1, 1, fit_stats=fit_stats, **fit_kwargs)

    if fit_marginal_x is not None:
        fit(fig, fit_marginal_x, 2, 1, 1, fit_stats=fit_stats)

    if fit_marginal_y is not None:
        fit(fig, fit_marginal_y, 1, 2, 1, fit_stats=fit_stats)

    return fig


def histogram(x, fit_type=None, fit_stats=True, umlf=False, fit_kwargs={}, **kwargs):
    fig = pux.histogram(x=x, **kwargs)

    if fit_type is not None:
        if umlf is False:
            result = standard_fit.fit(fig.data[0].x, fig.data[0].y, fit_type, **fit_kwargs)
        else:
            from standard_fit import unbinned_maximum_likelihood_fit
            result = unbinned_maximum_likelihood_fit.fit(x, fit_type, **fit_kwargs)
        fig._fit_result = result
        fig.add_trace(util.get_fit_trace(result, fig.data[0].x))
        if fit_stats:
            util.add_annotation(fig, result)
    return fig
