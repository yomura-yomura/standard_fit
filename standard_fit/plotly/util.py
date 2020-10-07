import numpy as np
import plotly.graph_objs as go
from standard_fit import functions
from uncertainties import ufloat


__all__ = ["get_fit_trace", "add_annotation"]


def get_fit_trace(result, n_points=1000, log_x=False, **kwargs):
    fit_type, params, _, _, _, x_range = result
    if log_x:
        fit_x = np.logspace(*np.log10(x_range), n_points)
    else:
        fit_x = np.linspace(*x_range, n_points)
    fit_y = [functions.get_func(fit_type)(x, *result[1]) for x in fit_x]

    plot_kwargs = dict(
        name=f"{fit_type} fit",
        line=dict(
            color="red"
        )
    )
    plot_kwargs.update(kwargs)

    return go.Scatter(mode="lines", x=fit_x, y=fit_y, **plot_kwargs)


def add_annotation(fig, fit_results):
    fit_type, params, cov_params, chi_squared, ndf, x_range = fit_results

    err_params = [np.sqrt(cov_params[i][i]) for i in range(len(params))]

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

    fig.add_annotation(
        # x=x0,
        # y=y0,
        x=fig.layout.xaxis.domain[1],
        y=fig.layout.yaxis.domain[1],
        # y=1,
        xref="paper",
        yref="paper",
        font=dict(
            # size=20
            size=10
        ),
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        align="left",
        showarrow=False,
        text="<br>".join(text_lines)
    )



