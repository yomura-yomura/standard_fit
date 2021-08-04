from .. import util
from ... import standard_fit as sf
import plotly_utility
import numpy as np
import itertools
import numpy_utility as npu


__all__ = ["fit"]


def fit(fig, fit_type, row=None, col=None, i_data=None, fit_stats=True, add_trace=True,
        datetime_type=None,
        use_all_data_as_one_in_one_trace=False,
        fit_kwargs=None, annotation_kwargs=None,
        fit_plot_kwargs=None):
    assert i_data is None

    if row == "all" or col == "all" or npu.is_array(row) or npu.is_array(col):
        n_row, n_col, n_data, *_ = np.shape(fig._grid_ref)
        if row == "all" or col == "all":
            if row == "all":
                row = list(range(1, n_row+1))
            if col == "all":
                col = list(range(1, n_col+1))
            row, col = zip(*itertools.product(row, col))

        if not npu.is_array(row):
            row = [row]
        if not npu.is_array(col):
            col = [col]

        # i_data = [i_data] if i_data != "all" else list(range(1, n_data+1))
        for r, c in zip(row, col):
            if fit_kwargs is not None and "print_result" in fit_kwargs and fit_kwargs["print_result"] == np.True_:
                print(f"\n* row={r}, col={c}")
            fit(
                fig, fit_type, r, c, None, fit_stats, add_trace,
                datetime_type,
                use_all_data_as_one_in_one_trace,
                fit_kwargs, annotation_kwargs, fit_plot_kwargs
            )
        return fig

    if row is None and fig._has_subplots():
        row = 1
    if col is None and fig._has_subplots():
        col = 1

    traces = plotly_utility.get_traces_at(fig, row, col)

    if len(traces) == 0:
        return fig

    plot_types_2d = ("scatter", "scattergl", "bar")
    plot_types_3d = ("scatter3d",)

    if use_all_data_as_one_in_one_trace:
        trace = traces[0]
        if not all(trace.type == t.type for t in traces[1:]):
            raise NotImplementedError("different trace types detected")
        assert trace.type in plot_types_2d

        if trace.type == "bar":
            assert all(trace.orientation == t.orientation for t in traces[1:])

            if trace.orientation == "h":
                x = "y"
                y = "x"
            else:
                x = "x"
                y = "y"

            bins = traces[0][x]
            if not np.all(np.all(bins == trace[x]) for trace in traces[1:]):
                raise NotImplementedError("different bins detected")
            counts = np.sum([trace[y] for trace in traces], axis=0)

            if trace.orientation == "h":
                y = bins
                x = counts
            else:
                x = bins
                y = counts

            x_err2 = [trace.error_x.array ** 2 for trace in traces if trace.error_x.array is not None]
            if len(x_err2) == len(traces):
                x_err = np.sqrt(np.sum(x_err2, axis=0))
            else:
                assert len(x_err2) == 0
                x_err = None

            y_err2 = [trace.error_y.array ** 2 for trace in traces if trace.error_y.array is not None]
            if len(y_err2) == len(traces):
                y_err = np.sqrt(np.sum(y_err2, axis=0))
            else:
                assert len(y_err2) == 0
                y_err = None
        else:
            x = [x for trace in traces for x in trace.x]
            y = [y for trace in traces for y in trace.y]
            x_err = [err_x
                     for trace in traces if trace.error_x.array is not None
                     for err_x in trace.error_x.array]
            if len(x_err) != len(x):
                assert len(x_err) == 0
                x_err = None

            y_err = [err_y
                     for trace in traces if trace.error_y.array is not None
                     for err_y in trace.error_y.array]
            if len(y_err) != len(x):
                print(y_err)
                assert len(y_err) == 0
                y_err = None
    else:
        i_data = 1
        if len(traces) < i_data:
            raise ValueError(f"i_data is specified as {i_data} but len(traces)={len(traces)} < i_data")
        trace = traces[i_data - 1]

        if trace.type in plot_types_2d:
            x = trace.x
            y = trace.y
            x_err = trace.error_x.array
            y_err = trace.error_y.array

        elif trace.type in plot_types_3d:
            x = np.c_[trace.x, trace.y]
            if trace.error_x.array is None or trace.error_y.array is None:
                x_err = None
            else:
                x_err = np.c_[trace.error_x.array, trace.error_y.array]

            y = trace.z
            y_err = trace.error_z.array
        else:
            raise NotImplementedError(f"{type(trace)} not supported.")

    if trace.type in plot_types_2d:
        log_x = fig.layout.xaxis.type == "log"
        flip_xy = trace.type == "bar" and trace.orientation == "h"

        if flip_xy:
            x, y = y, x
            x_err, y_err = y_err, x_err
    else:
        log_x = False
        flip_xy = False

    # if np.issubdtype(x.dtype, np.datetime64) or np.issubdtype(y.dtype, np.datetime64):
    #     if np.issubdtype(x.dtype, np.datetime64):
    #         if datetime_type is not None:
    #             x = x.astype(f"M8[{datetime_type}]")
    #         x_ = (x - x.min()).astype(int)
    #     else:
    #         x_ = x
    #
    #     if np.issubdtype(y.dtype, np.datetime64):
    #         if datetime_type is not None:
    #             y = y.astype(f"M8[{datetime_type}]")
    #         y_ = (y - y.min()).astype(int)
    #         assert y_err is None
    #     else:
    #         y_ = y
    #
    #     result = sf.fit(x_, y_, fit_type, error_x=x_err, error_y=y_err, **fit_kwargs)
    #     if result is None:
    #         return fig
    #
    #     if add_trace:
    #         fit_trace = util.get_fit_trace(result, x_, log_x=log_x, flip_xy=flip_xy, showlegend=False, **fit_plot_kwargs)
    #         fit_trace.x = fit_trace.x.astype(int) + x.min()
    #         fig.add_trace(fit_trace, row, col)
    #     if fit_stats:
    #         util.add_annotation(fig, result, row, col, **annotation_kwargs)
    #         time_unit = "".join(str(e) for e in reversed(np.datetime_data(x.dtype)))
    #         fig.layout.annotations[-1].text += "".join([
    #             "<br>",
    #             f"(starts from {x.min()}", "<br>",
    #             f" with time unit {time_unit})"
    #         ])
    # else:
    # if True:
    if fit_kwargs is None:
        fit_kwargs = dict()

    result = sf.fit(x, y, fit_type, error_x=x_err, error_y=y_err, **fit_kwargs)
    if result is None:
        return fig

    if add_trace:
        if fit_plot_kwargs is None:
            fit_plot_kwargs = dict()
        fig.add_trace(
            util.get_fit_trace(result, x, log_x=log_x, flip_xy=flip_xy, showlegend=False, **fit_plot_kwargs),
            row, col
        )

    if fit_stats:
        if annotation_kwargs is None:
            annotation_kwargs = dict()
        util.add_annotation(fig, result, row, col, **annotation_kwargs)

    # print(result)
    result = np.array(
        result,
        dtype=sf.get_fit_result_dtype(sf.regression.get_parameter_names(fit_type, result[-1]), 2 if result[-1] else 1)
    )

    if not hasattr(fig, "_fit_results"):
        fig._fit_results = dict()

    if row is None and col is None:
        if fit_type not in fig._fit_results:
            fig._fit_results[fit_type] = result[np.newaxis]
        else:
            fig._fit_results[fit_type] = np.concatenate((fig._fit_results[fit_type], result[np.newaxis]))
    else:
        if fit_type not in fig._fit_results:
            n_row, n_col = (e.stop - 1 for e in fig._get_subplot_rows_columns())
            fig._fit_results[fit_type] = np.ma.zeros((n_row, n_col, 1), dtype=result.dtype)
            fig._fit_results[fit_type].mask = True
        else:
            if fig._fit_results[fit_type][row-1][col-1][-1].mask["fit_type"] == np.True_:
                pass
            else:
                fig._fit_results[fit_type] = npu.pad(fig._fit_results[fit_type], [(0, 0), (0, 0), (0, 1)])
        fig._fit_results[fit_type][row - 1][col - 1][-1] = result

    return fig
