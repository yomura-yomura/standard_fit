import numpy as np
import plotly_utility.express as pux
import plotly_utility
import scipy.stats


__all__ = ["get_p_values", "get_chi2_dists"]


def get_p_values(fit_results):
    if isinstance(fit_results, np.ma.MaskedArray):
        p_values = np.ma.empty(fit_results.shape)
        mask = fit_results["fcn"].mask | fit_results["ndf"].mask
        p_values[~mask] = get_p_values(fit_results.data[~mask])
        p_values.mask = mask
        return p_values

    return scipy.stats.chi2.sf(fit_results["fcn"], fit_results["ndf"])


def get_chi2_dists(fit_results, facet_col_wrap=4, upper_chi2=None):
    if isinstance(fit_results, np.ma.MaskedArray):
        fit_results = fit_results.data[~(fit_results["fcn"].mask | fit_results["ndf"].mask)]
    if upper_chi2 is None:
        upper_chi2 = fit_results["ndf"].max() * 5

    fit_results = fit_results[(fit_results["fcn"] < upper_chi2) & (fit_results["ndf"] > 0)]

    fig = pux.histogram(
        fit_results[["ndf", "fcn"]],
        x="fcn", facet_col="ndf", facet_col_wrap=facet_col_wrap,
        histnorm="probability density",
        labels={"fcn": "χ²"}
    )

    fig_data, fig_coords = plotly_utility.to_numpy(fig, return_coords=True)

    for ndf, col, row, i_data in zip(
        fig_data["facet_col"].compressed(),
        *(
            item[~fig_data["facet_col"].mask]
            for item in np.meshgrid(fig_coords["column"], fig_coords["row"][::-1], fig_coords["trace"])
        )
    ):
        if int(ndf) < 1:
            continue
        x = np.linspace(0, upper_chi2, 100)
        fig.add_trace(dict(
            name=f"chi2 (ndf={ndf})",
            x=x,
            y=scipy.stats.chi2.pdf(x, int(ndf)),
            line_color="red"
        ), row=row + 1, col=col + 1)

    return fig
