from .fit_functions import fit
import plotly_utility.express as pux
import plotly.express as px


__all__ = ["scatter", "scatter_3d"]


def scatter(df=None, x=None, y=None, fit_type=None, fit_stats=True, fit_kwargs=None,
            fit_marginal_x=None, fit_marginal_y=None, annotation_kwargs=None,
            **kwargs):
    if fit_marginal_x is not None:
        kwargs["marginal_x"] = "histogram"
    if fit_marginal_y is not None:
        kwargs["marginal_y"] = "histogram"

    fig = pux.scatter(df, x=x, y=y, **kwargs)

    if fit_type is not None:
        fit(fig, fit_type, 1, 1, fit_stats=fit_stats, annotation_kwargs=annotation_kwargs, fit_kwargs=fit_kwargs)

    if fit_marginal_x is not None:
        fit(fig, fit_marginal_x, 2, 1, fit_stats=fit_stats, annotation_kwargs=annotation_kwargs)

    if fit_marginal_y is not None:
        fit(fig, fit_marginal_y, 1, 2, fit_stats=fit_stats, annotation_kwargs=annotation_kwargs)

    return fig


def scatter_3d(df=None, x=None, y=None, z=None, error_z=None, fit_type=None, fit_stats=True, fit_kwargs={},
               marker_size=1,
               annotation_kwargs={}, **kwargs):

    fig = px.scatter_3d(df, x=x, y=y, z=z, error_z=error_z, **kwargs).update_traces(marker_size=marker_size)

    if fit_type is not None:
        fit(fig, fit_type, 1, 1, fit_stats=fit_stats, annotation_kwargs=annotation_kwargs, **fit_kwargs)

    return fig
