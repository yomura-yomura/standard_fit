import plotly.express as px
from standard_fit.plotly import util
import plotly_utility.express as pux
from standard_fit.fit import fit


__all__ = ["scatter", "histogram"]


def scatter(x, y, fit_type=None, fit_stats=True, fit_kwargs={}, **kwargs):
    fig = px.scatter(x=x, y=y, **kwargs)
    if fit_type is not None:
        result = fit(fig.data[0].x, fig.data[0].y, fit_type, **fit_kwargs)
        fig._fit_result = result
        if "log_x" in kwargs:
            fig.add_trace(util.get_fit_trace(result, log_x=True), 1, 1)
        else:
            fig.add_trace(util.get_fit_trace(result), 1, 1)
        if fit_stats:
            util.add_annotation(fig, result)
    return fig


def histogram(x, do_fit=True, fit_stats=True, fit_kwargs={}, **kwargs):
    fig = pux.histogram(x=x, **kwargs)
    if do_fit:
        result = fit(fig.data[0].x, fig.data[0].y, "gaussian", **fit_kwargs)
        fig._fit_result = result
        fig.add_trace(util.get_fit_trace(result))
        if fit_stats:
            util.add_annotation(fig, result)
    return fig