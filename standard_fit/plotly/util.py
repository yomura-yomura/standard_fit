import numpy as np
import plotly.graph_objs as go
from .. import regression
import io
import PIL.Image
import PIL.ImageOps
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools


mpl.rcParams['text.latex.preamble'] = r"".join([
    r"\usepackage{{amsmath}}"
])


class ufloat:
    def __init__(self, nominal_value, std_dev, significant_digits=4, use_plus_zero_instead_of_minus=True):
        if std_dev < 0:
            raise ValueError("The standard deviation cannot be negative")

        self.nominal_value = nominal_value
        self.std_dev = std_dev
        self.significant_digits = significant_digits

        if use_plus_zero_instead_of_minus:
            if self.nominal_value == 0:
                self.nominal_value = 0
            if self.std_dev == 0:
                self.std_dev = 0

        if np.isfinite(self.nominal_value) and np.isfinite(self.std_dev):
            target = max(abs(self.nominal_value), self.std_dev)
        elif np.isfinite(self.nominal_value) and not np.isfinite(self.std_dev):
            target = abs(self.nominal_value)
        elif not np.isfinite(self.nominal_value) and np.isfinite(self.std_dev):
            target = self.std_dev
        else:
            target = 1

        if target == 0:
            target = 1

        self.factor = int(np.floor(np.log10(target)))

        if abs(self.factor) >= significant_digits:
            self.nominal_value /= 10 ** self.factor
            self.std_dev /= 10 ** self.factor

        factor = int(np.floor(np.log10(max(abs(self.nominal_value), self.std_dev))))
        self.valid_decimals = (
            self.significant_digits - 1
            - np.sign(factor) * (abs(factor) % self.significant_digits)
         )
        # self.valid_decimals = (
        #     self.significant_digits - np.sign(self.factor) * (abs(self.factor) % self.significant_digits) - 1
        # )

    def to_latex(self):
        s = rf"{self.nominal_value:.{self.valid_decimals}f} \pm {self.std_dev:.{self.valid_decimals}f}"

        if abs(self.factor) < self.significant_digits:
            return s
        else:
            return rf"({s}) \times 10^{{{self.factor}}}"

    def __str__(self):
        s = f"{self.nominal_value:.{self.valid_decimals}f} +/- {self.std_dev:.{self.valid_decimals}f}"

        if abs(self.factor) < self.significant_digits:
            return s
        else:
            return f"({s})e{self.factor}"

    def __repr__(self):
        return f"ufloat({self.__str__()})"


__all__ = ["get_fit_trace", "add_annotation"]


def get_fit_trace(result, x, n_points=None, log_x=False, flip_xy=False, showlegend=False):
    fit_type, params, _, _, _, x_range, y_range, *_, is_multivariate = result

    x = np.asarray(x)
    # is_multivariate = x.ndim == 2
    
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

    fit_y = regression.eval(fit_x, result)
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
        dpi=150,
        # use_font_size=False,
        # font_size=40,
        # annotation_family="Arial",
        max_occupied_ratio=0.25,
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

    param_names = regression.get_parameter_names(fit_type, is_multivariate)

    def cast(s):
        table = {
            "μ": r"\mu",
            "Σ": r"\Sigma",
            "σ": r"\sigma",
            "τ": r"\tau",
            "ψ": r"\psi",
        }

        for symbol, latex_symbol in table.items():
            if symbol in s:
                s = s.replace(symbol, latex_symbol)

        if "_" in s:
            s = s.replace("_", r"\_")

        return s

    def to_latex_power(s):
        if "e" in s:
            sig, index = s.split("e")
            return rf"{sig} \times 10^{{{index.lstrip('+0')}}}"
        return s

    chi2 = f"{chi_squared:.{valid_digits}g}"
    text_lines = [
        fr"\chi^2/\mathrm{{ndf}} &= {to_latex_power(chi2)}/{ndf}"
    ]
    if display_matrix:

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
                indices = a[0][0]
            elif ndim == 1:
                indices = a[:, 0]
                order = a[:, 1]
                indices = indices[order]
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
            else:
                raise NotImplementedError

            return np.vectorize(lambda i: ufloat(params[i], err_params[i], valid_digits).to_latex())(indices)

        def to_latex(s):
            def to_latex_matrix(p):
                if p.ndim == 0:
                    return p
                elif p.ndim == 1:
                    return " & ".join(p)
                elif p.ndim == 2:
                    return r" \\ ".join(to_latex_matrix(ip) for ip in p)
                else:
                    assert False

            if s.ndim == 0:
                return s
            else:
                return (
                    rf"\begin{{bmatrix}} {to_latex_matrix(s)} \end{{bmatrix}}"
                )

        text_lines.extend([
            f"{cast(pn)} &= {to_latex(make_matrix(make_array(list(r))))}"
            for pn, r in itertools.groupby(
                ((i, *pn.split("_")) for i, pn in enumerate(param_names)), lambda r: r[1]
            )
        ])
    else:
        text_lines.extend([
            f"{cast(pn)} &= {ufloat(p, ep, valid_digits).to_latex()}"
            for pn, p, ep in zip(param_names, params, err_params)
        ])

    if isinstance(fit_result, np.ndarray) and "p-value" in fit_result.dtype.names:
        text_lines.append(f"Probability &= {fit_result['p-value']:.2f}")

    text = r"\begin{{align*}} {} \end{{align*}}".format(r"\\".join(text_lines))

    plt_fig = plt.figure(figsize=(6.4, 4.8 * int(np.ceil(len(text_lines) / 10))), dpi=dpi)
    plt_fig.text(
        x=0.5, y=0.5,
        s=text,
        fontsize=20, va="center", ha="center", usetex=True
    )

    with io.BytesIO() as fp:
        plt_fig.savefig(fp, format="png")
        img_a = np.asarray(PIL.Image.open(fp))
    non_zeros_mask = ~np.all(img_a == 255, axis=-1)
    rows, cols = np.where(non_zeros_mask)
    margin_x = margin_y = 20

    assert rows.min() >= margin_y
    assert rows.max() <= img_a.shape[0] + margin_y
    assert cols.min() >= margin_x
    assert cols.max() <= img_a.shape[1] + margin_x
    cropped_img_a = img_a[
        rows.min() - margin_y:rows.max() + margin_y,
        cols.min() - margin_x:cols.max() + margin_x
    ]

    cropped_img_with_boarders = PIL.ImageOps.expand(
        PIL.Image.fromarray(cropped_img_a),
        border=4, fill="#c7c7c7"
    )

    x_per_y = cropped_img_with_boarders.height / cropped_img_with_boarders.width * (x1 - x0) / (y1 - y0)
    width = max_occupied_ratio * 2 * (x1 - x0)
    height = width * x_per_y

    if y1 - y0 < height:
        height = y1 - y0
        width = height / x_per_y
        max_occupied_ratio = width / 2

    if inside:
        xanchor = "right"
    else:
        xanchor = "left"
        x1 -= (x1 - x0) * max_occupied_ratio
        subplot.xaxis.domain = (x0, x1)

    fig.add_layout_image(
        # xref="x domain", yref="y domain",
        xref="paper", yref="paper",
        xanchor=xanchor,
        yanchor="top",
        x=x1, y=y1,
        sizex=width,
        sizey=height,
        source=cropped_img_with_boarders,
        # row=row, col=col
    )

    return fig

