import plotly_utility.express as pux
from .. import util
from ... import standard_fit


__all__ = ["histogram"]


def histogram(df=None, x=None, fit_type=None, fit_stats=True, umlf=False, fit_kwargs={}, annotation_kwargs={},
              **kwargs):
    fig = pux.histogram(df, x=x, **kwargs)

    if fit_type is not None:
        if umlf is False:
            result = standard_fit.fit(fig.data[0].x, fig.data[0].y, fit_type, **fit_kwargs)
        else:
            from standard_fit import unbinned_maximum_likelihood_fit
            result = unbinned_maximum_likelihood_fit.fit(x, fit_type, **fit_kwargs)
        fig._fit_result = result
        fig.add_trace(util.get_fit_trace(result, fig.data[0].x))
        if fit_stats:
            util.add_annotation(fig, result, **annotation_kwargs)

    return fig