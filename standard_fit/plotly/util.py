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

        factor = int(np.floor(np.log10(target)))
        # self.valid_decimals = (
        #     self.significant_digits - np.sign(self.factor) * (abs(self.factor) % self.significant_digits) - 1
        # )
        # self.valid_decimals = (
        #     self.significant_digits - 1
        #     - np.sign(factor) * (abs(factor) % self.significant_digits)
        # )
        if 0 <= self.factor < 4:
            self.valid_decimals = self.significant_digits - 1 - factor % self.significant_digits
        else:
            self.valid_decimals = self.significant_digits - 1

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


def get_fit_trace(result, x, n_points=None, log_x=False, flip_xy=False, showlegend=False, fit_x_range=None):
    if isinstance(result, np.ndarray):
        fit_type = result["fit_type"]
        # params = result["params"]
        x_range = result["x_range"]
        y_range = result["y_range"]
        is_multivariate = result["is_multivariate"]
    else:
        fit_type, params, _, _, _, x_range, y_range, *_, is_multivariate = result

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.isfinite(x)]
    elif x.ndim == 2:
        x = x[np.isfinite(x).any(axis=-1)]
    else:
        raise NotImplementedError

    if is_multivariate:
        if n_points is None:
            n_points = int(np.power(len(x) * 20, 1.5))

        assert log_x is False
        n_variables = x.shape[1]
        if 2 < n_variables:
            raise NotImplementedError("2 < n_variables")

        # x_margin = 0.1 * (x.max(axis=0) - x.min(axis=0))
        # print(x_margin)
        x_margin = [0, 0]
        if fit_x_range is None:
            fit_x_range = [
                (max(x_range[0][0], x[:, 0].min() - x_margin[0]), min(x_range[0][1], x[:, 0].max() + x_margin[0])),
                (max(x_range[1][0], x[:, 1].min() - x_margin[1]), min(x_range[1][1], x[:, 1].max() + x_margin[1]))
            ]

        # fit_x = np.stack([
        #     e.flatten()
        #     for e in np.meshgrid(
        #         np.linspace(fit_x_range[0][0], fit_x_range[0][1], int(np.sqrt(n_points))),
        #         np.linspace(fit_x_range[1][0], fit_x_range[1][1], int(np.sqrt(n_points)))
        #     )
        # ], axis=-1)
        fit_x = (
            np.linspace(fit_x_range[0][0], fit_x_range[0][1], int(np.sqrt(n_points))),
            np.linspace(fit_x_range[1][0], fit_x_range[1][1], int(np.sqrt(n_points)))
        )
    else:
        if n_points is None:
            n_points = min(len(x) * 20, 300_000)

        if log_x is True:
            x_margin = 0.1 * (np.log10(np.max(x)) - np.log10(np.min(x)))
            if fit_x_range is None:
                fit_x_range = (
                    max(x_range[0], 10 ** (np.log10(np.min(x)) - x_margin)),
                    min(x_range[1], 10 ** (np.log10(np.max(x)) + x_margin))
                )
            fit_x = np.logspace(*np.log10(fit_x_range), n_points)
        else:
            x_margin = 0.1 * (np.max(x) - np.min(x))
            if fit_x_range is None:
                fit_x_range = (
                    max(x_range[0], np.min(x) - x_margin),
                    min(x_range[1], np.max(x) + x_margin)
                )
            fit_x = np.linspace(*fit_x_range, n_points)

    if is_multivariate:
        fit_y = regression.eval(
            np.stack([
                e.flatten()
                for e in np.meshgrid(fit_x[0], fit_x[1])
            ], axis=-1),
            result
        )
    else:
        fit_y = regression.eval(fit_x, result)
    matched_on_y = (y_range[0] <= fit_y) & (fit_y <= y_range[1])
    fit_y[~matched_on_y] = np.nan
    # if is_multivariate:
    #     fit_x[0][~matched_on_y] = fit_x[1][~matched_on_y] = np.nan
    # else:
    #     fit_x[~matched_on_y] = np.nan

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
            # x=np.linspace(x[:, 0].min(), x[:, 0].max(), int(np.sqrt(n_points))),
            # y=np.linspace(x[:, 1].min(), x[:, 1].max(), int(np.sqrt(n_points))),
            x=fit_x[0],
            y=fit_x[1],
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
        max_size_x=0.25, max_size_y=0.5,
        valid_digits=4,
        display_matrix=False,
        position="top right",
        use_latex_conversion=False,
        param_name_conversion_table: dict = None
):
    assert i_data == 1  # not implemented yet
    fit_type, params, err_params, chi_squared, ndf, *_, is_multivariate = fit_result
    if isinstance(fit_result, np.ndarray) and "p-value" in fit_result.dtype.names:
        p_value = fit_result["p-value"]
    else:
        p_value = None

    _add_annotation(
        fig, fit_type, params, err_params, chi_squared, ndf, is_multivariate, p_value,
        row, col,
        inside,
        dpi,
        max_size_x, max_size_y,
        valid_digits,
        display_matrix,
        position,
        use_latex_conversion,
        param_name_conversion_table
    )


def _add_annotation(
    fig, fit_type, params, err_params, chi_squared, ndf, is_multivariate, p_value=None,
    row=1, col=1,
    inside=True,
    dpi=150,
    max_size_x=0.25, max_size_y=0.5,
    valid_digits=4,
    display_matrix=False,
    position="top right",
    use_latex_conversion=False,
    param_name_conversion_table: dict = None
):
    if not fig._has_subplots() or (row is None and col is None):
        subplot = fig.layout  # not subplot in this case
        x0 = y0 = 0.
        x1 = y1 = 1.
    else:
        subplot = fig.get_subplot(row, col)
        if is_multivariate:
            x0, x1 = subplot.domain["x"]
            y0, y1 = subplot.domain["y"]
        else:
            x0, x1 = subplot.xaxis.domain
            y0, y1 = subplot.yaxis.domain

    if hasattr(params, "dtype"):
        param_names = params.dtype.names
    else:
        param_names = regression.get_parameter_names(fit_type, is_multivariate)

    if param_name_conversion_table is not None:
        param_name_conversion_table = {
            k: v[1:-1] if v.startswith("$") and v.endswith("$") else v
            for k, v in param_name_conversion_table.items()
        }
        param_names = [param_name_conversion_table.get(name, name) for name in param_names]

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

        if use_latex_conversion == np.False_:
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
                sorted(((i, *pn.split("_")) for i, pn in enumerate(param_names)), key=lambda r: r[1]),
                key=lambda r: r[1]
            )
        ])
    else:
        text_lines.extend([
            f"{cast(pn)} &= {ufloat(p, ep, valid_digits).to_latex()}"
            for pn, p, ep in zip(param_names, params, err_params)
        ])

    if p_value is not None:
        text_lines.append(f"Probability &= {p_value:.2f}")

    def get_image(text_lines, dpi):
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
        plt.close(plt_fig)

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
        return cropped_img_with_boarders

    image = get_image(text_lines, dpi)

    width = max_size_x * (x1 - x0)
    height = max_size_y * (y1 - y0)

    attrs = position.split()
    if len(attrs) != 2:
        raise ValueError("annotation_position must be a text including top/bottom and right/left separated with a space")

    if inside:
        xanchor = "right" if "right" in attrs else "left"
    else:
        if "right" in attrs:
            xanchor = "left"
            x1 -= width
        else:
            xanchor = "right"
            x0 += width

        if x1 <= x0:
            raise ValueError("max_occupied_ratio < 1 if inside == False")

        subplot.xaxis.domain = (x0, x1)

    if "top" in attrs:
        y = y1
        yanchor = "top"
    elif "bottom" in attrs:
        y = y0
        yanchor = "bottom"
    else:
        raise ValueError("top/bottom should be specified in annotation_position")

    if "right" in attrs:
        x = x1
    elif "left" in attrs:
        x = x0
    else:
        raise ValueError("right/left should be specified in annotation_position")

    fig.add_layout_image(
        # xref="x domain", yref="y domain",
        # row=row, col=col
        xref="paper", yref="paper",
        xanchor=xanchor,
        yanchor=yanchor,
        x=x, y=y,
        sizex=width,
        sizey=height,
        source=image
    )

    return fig

