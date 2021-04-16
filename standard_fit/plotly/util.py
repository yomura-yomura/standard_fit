import numpy as np
import plotly.graph_objs as go
from standard_fit import regression
from uncertainties import ufloat
import warnings
import tkinter as tk
import tkinter.font


root = tk.Tk()
root.withdraw()


__all__ = ["get_fit_trace", "add_annotation"]


def get_fit_trace(result, x, n_points=None, log_x=False, flip_xy=False, showlegend=False):
    fit_type, params, _, _, _, x_range, y_range, *_ = result

    x = np.asarray(x)
    is_multivariate = x.ndim == 2
    
    if is_multivariate:
        if n_points is None:
            n_points = 10_000

        assert log_x is False
        n_variables = x.shape[1]
        if 2 < n_variables:
            raise NotImplementedError("2 < n_variables")

        fit_x = np.stack([
            e.flatten()
            for e in np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max(), int(np.sqrt(n_points))),
                                 np.linspace(x[:, 1].min(), x[:, 1].max(), int(np.sqrt(n_points))))
        ], axis=-1)
    else:
        if n_points is None:
            n_points = 1_000

        if log_x is True:
            x_margin = 0.1 * (np.log10(np.max(x)) - np.log10(np.min(x)))
            fit_x_range = (max(x_range[0], 10 ** (np.log10(np.min(x)) - x_margin)),
                           min(x_range[1], 10 ** (np.log10(np.max(x)) + x_margin)))
            fit_x = np.logspace(*np.log10(fit_x_range), n_points)
        else:
            x_margin = 0.1 * (np.max(x) - np.min(x))
            fit_x_range = (max(x_range[0], np.min(x) - x_margin),
                           min(x_range[1], np.max(x) + x_margin))
            fit_x = np.linspace(*fit_x_range, n_points)

    fit_y = regression.eval(fit_type, fit_x, result[1])
    matched_on_y = (y_range[0] <= fit_y) & (fit_y <= y_range[1])
    fit_x[~matched_on_y] = fit_y[~matched_on_y] = np.nan

    plot_kwargs = dict(
        name=f"{fit_type} fit",
        showlegend=showlegend
    )

    if is_multivariate:
        assert flip_xy is False

        import plotly.express as px
        # plot_kwargs.update(color="#EF553B", opacity=0.3)
        # return go.Mesh3d(x=fit_x[:, 0], y=fit_x[:, 1], z=fit_y, **plot_kwargs)
        # plot_kwargs.update(colorscale=px.colors.sequential.Reds[2:], opacity=0.5, showscale=False)
        plot_kwargs.update(colorscale=["#EF553B"] * 2, opacity=0.3, showscale=False)
        return go.Surface(
            x=np.linspace(x[:, 0].min(), x[:, 0].max(), int(np.sqrt(n_points))),
            y=np.linspace(x[:, 1].min(), x[:, 1].max(), int(np.sqrt(n_points))),
            z=fit_y.reshape((int(np.sqrt(n_points))), int(np.sqrt(n_points))),
            **plot_kwargs
        )
    else:
        if flip_xy:
            fit_x, fit_y = fit_y, fit_x

        plot_kwargs.update(line=dict(color="red"))
        return go.Scattergl(mode="lines", x=fit_x, y=fit_y, **plot_kwargs)


def add_annotation(
        fig, fit_result, row=1, col=1, i_data=1,
        inside=True,
        use_font_size=False,
        font_size=40,
        annotation_family="Arial",
        occupied_ratio=0.25,
        valid_digits=4,
        display_matrix=False
):
    assert i_data == 1  # not implemented yet
    fit_type, params, err_params, chi_squared, ndf, *_, is_multivariate = fit_result

    if not fig._has_subplots() or (row is None and col is None):
        subplot = fig.layout  # not subplot in this case
        x0 = 0.
        x1 = y1 = 1.
    else:
        subplot = fig.get_subplot(row, col)
        if is_multivariate:
            x0, x1 = subplot.domain["x"]
            y0, y1 = subplot.domain["y"]
        else:
            x0, x1 = subplot.xaxis.domain
            y0, y1 = subplot.yaxis.domain

    aspect_ratio = root.winfo_screenheight() / root.winfo_screenwidth()
    if fig.layout.width is None:
        fig.layout.width = root.winfo_screenwidth() * 0.85
    if fig.layout.height is None:
        fig.layout.height = fig.layout.width * aspect_ratio

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

    param_names = regression.get_parameter_names(fit_type, is_multivariate)

    if display_matrix:
        import itertools

        def make_array(a):
            shape = np.shape(a)
            if len(shape) != 2:
                raise ValueError(a)
            ndim = shape[1] - 2
            if not (0 <= ndim <= 2):
                raise ValueError("0 <= ndim <= 2")
            return np.array([(r[0], *r[2:]) for r in a], dtype="i4")

        def make_matrix(a):
            ndim = a.shape[1] - 1
            if ndim == 0:
                assert len(a) == 1
                index = a[0][0]
                return np.take(str_p, index), np.take(str_ep, index)
            elif ndim == 1:
                indices = a[:, 0]
                order = a[:, 1]
                return np.take(str_p, indices)[order], np.take(str_ep, indices)[order]
            elif ndim == 2:
                m_row_indices = np.unique(a[:, 1])
                assert all(m_row_indices == np.arange(len(m_row_indices)))
                m_col_indices = np.unique(a[:, 2])
                assert all(m_col_indices == np.arange(len(m_col_indices)))
                indices = np.empty((len(m_row_indices), len(m_col_indices)), dtype="i8")
                if indices.size == len(a):
                    indices[a[:, 1], a[:, 2]] = a[:, 0]
                elif len(a) < indices.size:
                    if indices.shape[0] == indices.shape[1]:
                        indices[a[:, 1], a[:, 2]] = indices[a[:, 2], a[:, 1]] = a[:, 0]
                    else:
                        raise ValueError("the number of parameters are not sufficient for non-symmetry matrix")
                else:
                    raise ValueError("many parameters given for matrix")
                return np.take(str_p, indices), np.take(str_ep, indices)

        def to_latex(m):
            def to_latex_matrix(p):
                if p.ndim == 0:
                    return p
                elif p.ndim == 1:
                    return " & ".join(p)
                elif p.ndim == 2:
                    return r" \\ ".join(to_latex_matrix(ip) for ip in p)
                else:
                    assert False

            p, ep = m
            return (
                rf"\begin{{bmatrix}} {to_latex_matrix(p)} \end{{bmatrix}} \pm " +
                rf"\begin{{bmatrix}} {to_latex_matrix(ep)} \end{{bmatrix}}"
            )

        text_lines = [
            fr"χ²/\mathrm{{ndf}} &= {chi_squared:.{valid_digits}g}/{ndf}",
            *(
                f"{pn} &= {to_latex(make_matrix(make_array(list(r))))}"
                for pn, r in itertools.groupby(
                    ((i, *pn.split("_")) for i, pn in enumerate(param_names)), lambda r: r[1]
                )
            )
        ]
        text = r"$\begin{{align}} {} \end{{align}}$".format(r"\\".join(text_lines))
        scale = 1
    else:
        text_lines = [
            f"χ²/ndf = {chi_squared:.{valid_digits}g}/{ndf} ",
            *[
                f"{n} = {p} ± {ep} " for n, p, ep in zip(
                    as_str(param_names),
                    as_str(str_p),
                    as_str(str_ep, align="left")
                )
            ]
        ]
        text = "<br>".join(text_lines)

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
        text=text
    )
    return fig

