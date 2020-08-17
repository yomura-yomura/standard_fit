import plotly.express as px
from . import _plotly
from ._plotly_histogram import histogram as plotly_histogram


def scatter(x, y, result=None, fit_stats=True, **kwargs):
    fig = px.scatter(x=x, y=y, **kwargs)
    if result is not None:
        fig._fit_result = result
        if "log_x" in kwargs:
            fig.add_trace(_plotly.get_fit_trace(result, log_x=True), 1, 1)
        else:
            fig.add_trace(_plotly.get_fit_trace(result), 1, 1)
        if fit_stats:
            _plotly.add_annotation(fig, result)
    return fig


def histogram(x, result=None, fit_stats=True, **kwargs):
    fig = plotly_histogram(x, **kwargs)
    if result is not None:
        fig.add_trace(_plotly.get_fit_trace(result))
        if fit_stats:
            _plotly.add_annotation(fig, result)
    return fig
