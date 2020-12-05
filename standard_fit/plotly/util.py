import numpy as np
import plotly.graph_objs as go
from standard_fit import functions
from uncertainties import ufloat


__all__ = ["get_fit_trace", "add_annotation"]


def get_fit_trace(result, x, n_points=1000, log_x=False, flip_xy=False, **kwargs):
    fit_type, params, _, _, _, x_range, y_range = result

    x_margin = 0.1 * (x.max() - x.min())
    # y_margin = 0.1 * (y.max() - y.min())

    fit_x_range = (max(x_range[0], x.min() - x_margin), min(x_range[1], x.max() + x_margin))

    if log_x:
        fit_x = np.logspace(*np.log10(fit_x_range), n_points)
    else:
        fit_x = np.linspace(*fit_x_range, n_points)
    fit_y = np.array([functions.get_func(fit_type)(x, *result[1]) for x in fit_x])

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


def add_annotation(fig, fit_result, row=1, col=1, i_data=1,
                   # text_size=15
                   text_size=20
                   ):
    # fit_type, params, cov_params, chi_squared, ndf, *_ = fit_result
    fit_type, params, err_params, chi_squared, ndf, *_ = fit_result

    # err_params = [np.sqrt(cov_params[i][i]) for i in range(len(params))]

    def as_str(a, formats=None, align="right"):
        if formats is None:
            a = [f"{s}" for s in a]
        else:
            a = [f"{s:{f}}" for s, f in zip(a, formats)]
        max_len = max([len(s) for s in a])
        if align == "right":
            return [f"{s:>{max_len}}" for s in a]

    if fit_type == "kde":
        str_params = [f"{ufloat(p, ep):.4g}".replace("+/-", " ± ") for p, ep in zip(params[1:], err_params[1:])]
        text_lines = [
            f"χ²/ndf = {chi_squared:.4g}/{ndf}",
            *[f"{n} = {p}" for n, p in zip(as_str(functions.get_parameter_names(fit_type)[1:]), str_params)]
        ]
    else:
        str_params = [f"{ufloat(p, ep):.4g}".replace("+/-", " ± ") for p, ep in zip(params, err_params)]

        text_lines = [
            f"χ²/ndf = {chi_squared:.4g}/{ndf}",
            *[f"{n} = {p}" for n, p in zip(as_str(functions.get_parameter_names(fit_type)), str_params)]
        ]

    if row is None and col is None:
        x0 = y0 = 0.
        x1 = y1 = 1.
    else:
        subplot = fig.get_subplot(row, col)
        x0, x1 = subplot.xaxis.domain
        y0, y1 = subplot.yaxis.domain
    scale = min(x1 - x0, y1 - y0)

    if text_size * scale < 1:
        scale = 1 / text_size

    fig.add_annotation(
        # x=x0,
        # y=y0,
        x=x1,
        y=y1,
        # y=1,
        xanchor="right",
        yanchor="top",
        xref="paper",
        yref="paper",
        font=dict(
            size=text_size * scale
        ),
        bordercolor="#c7c7c7",
        borderwidth=2 * scale,
        borderpad=4 * scale,
        bgcolor="white",
        align="left",
        showarrow=False,
        text="<br>".join(text_lines)
    )



