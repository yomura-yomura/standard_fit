import numpy as np
import plotly.graph_objs as go
from standard_fit import regression
from uncertainties import ufloat
import warnings
import tkinter as tk
import tkinter.font


__all__ = ["get_fit_trace", "add_annotation"]


def get_fit_trace(result, x, n_points=1000, log_x=False, flip_xy=False, **kwargs):
    fit_type, params, _, _, _, x_range, y_range, *_ = result

    if log_x is True:
        x_margin = 0.1 * (np.log10(np.nanmax(x)) - np.log10(np.nanmin(x)))
        fit_x_range = (max(x_range[0], 10 ** (np.log10(np.nanmin(x)) - x_margin)),
                       min(x_range[1], 10 ** (np.log10(np.nanmax(x)) + x_margin)))
    else:
        x_margin = 0.1 * (np.nanmax(x) - np.nanmin(x))
        # y_margin = 0.1 * (y.max() - y.min())
        fit_x_range = (max(x_range[0], np.nanmin(x) - x_margin),
                       min(x_range[1], np.nanmax(x) + x_margin))

    if log_x:
        fit_x = np.logspace(*np.log10(fit_x_range), n_points)
    else:
        fit_x = np.linspace(*fit_x_range, n_points)

    fit_y = regression.eval(fit_type, fit_x, result[1])
    # if regression.linear_regression.is_defined(fit_type):
    # else:
    #     fit_y = np.array([nonlinear_regression.get_func(fit_type)(x, *result[1]) for x in fit_x])

    matched_on_y = (y_range[0] <= fit_y) & (fit_y <= y_range[1])
    fit_x[~matched_on_y] = fit_y[~matched_on_y] = np.nan

    plot_kwargs = dict(
        name=f"{fit_type} fit",
        line=dict(
            color="red"
        )
    )
    plot_kwargs.update(kwargs)

    if flip_xy:
        fit_x, fit_y = fit_y, fit_x

    return go.Scattergl(mode="lines", x=fit_x, y=fit_y, **plot_kwargs)


def add_annotation(
        fig, fit_result, row=1, col=1, i_data=1,
        inside=True,
        use_font_size=False,
        font_size=40,
        annotation_family="Arial",
        occupied_ratio=0.25,
        valid_digits=4
):
    assert i_data == 1  # not implemented yet
    fit_type, params, err_params, chi_squared, ndf, *_ = fit_result

    standard_height = 1000
    standard_width = 1600
    aspect_ratio = standard_height / standard_width  # 16:10

    tk.Frame().destroy()
    font = tk.font.Font(family="Arial", size=-font_size)

    ws = " "
    ws_width = font.measure(ws * 100) / 100

    def as_str(a, formats=None, align="right"):
        if formats is None:
            a = [f"{s}" for s in a]
        else:
            a = [f"{s:{f}}" for s, f in zip(a, formats)]

        max_w = max(font.measure(s) for s in a)
        if align == "right":
            return [ws * int((max_w - font.measure(s)) / ws_width) + s if font.measure(s) < max_w else s for s in a]
        elif align == "left":
            return [s + ws * int((max_w - font.measure(s)) / ws_width) if font.measure(s) < max_w else s for s in a]

    params = [0 if p == 0 else p for p in params]  # minus zeros to plus zeros

    str_p, str_ep = zip(*(
        f"{ufloat(p, ep):.{valid_digits}g}".split("+/-")
        if not (np.isnan(p) or np.isnan(ep)) else (f"{p:.{valid_digits}g}", f"{ep:.{valid_digits}g}")
        for p, ep in zip(params, err_params)
    ))

    text_lines = [
        f"χ²/ndf = {chi_squared:.{valid_digits}g}/{ndf} ",
        *[
            f"{n} = {p} ± {ep} " for n, p, ep in zip(
                as_str(regression.get_parameter_names(fit_type)),
                as_str(str_p),
                as_str(str_ep, align="left")
            )
          ]
    ]


    if not fig._has_subplots() or (row is None and col is None):
        subplot = fig.layout  # not subplot in this case
        x0 = 0.
        x1 = y1 = 1.
    else:
        subplot = fig.get_subplot(row, col)
        x0, x1 = subplot.xaxis.domain
        y0, y1 = subplot.yaxis.domain

    if fig.layout.width is None:
        fig.layout.width = standard_width
    if fig.layout.height is None:
        fig.layout.height = fig.layout.width * aspect_ratio

    if use_font_size is True:
        scale = 1
    else:
        h = len(text_lines) * font.metrics("linespace") + 20 * (len(text_lines) - 1)
        w = max(font.measure(l) for l in text_lines)
        if h > fig.layout.height:
            scale = fig.layout.height / h
        else:
            scale = 1

        scale = min(scale, fig.layout.width * (x1 - x0) * occupied_ratio / w)

        if font_size * scale < 1:
            warnings.warn("Calculated text size is set to 1 because text size attribute must be greater than 1.")
            scale = 1 / font_size

        if not inside:
            x1 -= (x1 - x0) * occupied_ratio
            subplot.xaxis.domain = (x0, x1 - 5 / fig.layout.width * scale)

    fig.add_annotation(
        x=x1,
        y=y1,
        xanchor="right" if inside else "left",
        yanchor="top",
        xref="paper",
        yref="paper",
        font=dict(
            family=annotation_family,
            size=font_size * scale
        ),
        bordercolor="#c7c7c7",
        borderwidth=2 * scale,
        borderpad=4 * scale,
        bgcolor="white",
        align="left",
        showarrow=False,
        text="<br>".join(text_lines)
    )


# def _estimate_range(x: list):
#     x_range = [min(x), max(x)]
#     x_margin = 0.06 * (x_range[1] - x_range[0])
#
#     x_range[0] -= x_margin
#     x_range[1] += x_margin
#     return x_range
#
#
# def set_estimated_range(fig):
#     if fig.layout.xaxis.range is None:
#         fig.update_xaxes(range=_estimate_range([x for trace in fig.data for x in trace.x]))
#     if fig.layout.yaxis.range is None:
#         fig.update_yaxes(range=_estimate_range([y for trace in fig.data for y in trace.y]))
#     return fig



