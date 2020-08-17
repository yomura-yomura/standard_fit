import numpy as np
import plotly.graph_objs as go


def generic_histogram(a, *args, **kwargs):
    def is_numeric(a: np.ndarray):
        return (
            np.issubdtype(a.dtype, np.signedinteger) or np.issubdtype(a.dtype, np.unsignedinteger) or
            np.issubdtype(a.dtype, np.floating)
        )

    a = np.array(a)
    assert a.ndim == 1
    # assert a.dtype != np.object
    import datetime as dt

    if is_numeric(a):
        counts, bins = np.histogram(a, *args, **kwargs)
        width = bins[1:] - bins[:-1]
        # x = 0.5 * (bins[1:] + bins[:-1])
    elif np.issubdtype(a.dtype, np.datetime64):
        if 0 < len(args) and np.issubdtype(np.array(args[0]).dtype, np.datetime64):
            args = ((args[0] - args[0].min()).astype(int), *args[1:])
        elif "bins" in kwargs and np.issubdtype(np.array(kwargs["bins"]).dtype, np.datetime64):
            kwargs["bins"] = (kwargs["bins"] - kwargs["bins"].min()).astype(int)

        counts, bins = np.histogram((a - a.min()).astype(int), *args, **kwargs)

        time_unit = np.datetime_data(a.dtype)[0]
        width = (bins[1:] - bins[:-1]).astype(f"timedelta64[{time_unit}]")
        # x = a.min() + (0.5*(bins[1:] + bins[:-1])).astype(int)
    else:
        import collections
        bins, counts = zip(*collections.Counter(a).items())
        bins = np.array(bins)
        width = np.array([1] * len(bins))
    # elif all([isinstance(t, dt.time) for t in a]):
    #
    # else:
    #     raise NotImplementedError

    return counts, bins, width


def _histogram(x, bins, density, opacity, name=None, color=None):
    if isinstance(bins, str):
        counts, bins, width = generic_histogram(x, bins=bins, density=density)
    else:
        counts, _, width = generic_histogram(x, bins=bins, density=density)

    # if np.issubdtype(bins.dtype, np.datetime64):
    if np.issubdtype(x.dtype, np.datetime64):
        delta_x = width.astype("timedelta64[us]")
        width = width.astype("timedelta64[us]").astype(int)
        # delta_x = width.astype(object).astype(str)
        # x = (bins[:-1] + width / 2).astype("datetime64[us]")
        x = (bins[:-1] + width / 2).astype("datetime64[us]")
    # elif np.issubdtype(bins.dtype, np.str_):
    elif np.issubdtype(x.dtype, np.str_):
        x = bins
        delta_x = [1] * len(x)
    else:
        delta_x = width / 2
        x = bins[:-1] + delta_x

    return go.Bar(
        name=name,
        x=x,
        y=counts,
        width=width,
        marker=dict(
            line=dict(
                width=0
            ),
            opacity=opacity,
            color=color
        ),
        legendgroup=name,
        showlegend=name is not None,

        # customdata=delta_x,
        # hovertemplate="x=%{x} ± %{customdata}<br>count=%{y}"
        customdata=np.c_[x-delta_x, x+delta_x],
        hovertemplate="x=%{customdata[0]} - %{customdata[1]}<br>count=%{y}"
    )


def histogram(x, marginal=None, color=None, barmode="stack",
              bins="auto", density=False,
              fit=None, **kwargs):
    import plotly.express as px

    if barmode == "overlay":
        opacity = 0.5
    else:
        opacity = 1

    x = np.array(x)
    try:
        mask = ~np.isnan(x) & ~np.isinf(x)
        x = x[mask]

        if color is not None:
            color = np.array(color)
            color = color[mask]
    except TypeError:
        pass

    fig = px.histogram(x=x, marginal=marginal, color=color, barmode=barmode, **kwargs)
    marginal_traces = [trace for trace in fig.data if not isinstance(trace, go.Histogram)]

    if color is None:
        data = [
            _histogram(x, bins, density, opacity)
        ]
    else:
        names = np.array(color)
        data = []

        bins = generic_histogram(x, bins=bins)[1]

        for marginal_trace in marginal_traces:
            name = marginal_trace.name
            data.append(
                _histogram(x[names == name], bins, density, opacity, name, marginal_trace.marker.color)
            )

        assert all([np.all(data[0].x == trace.x) for trace in data[1:]])

    if fit is not None:
        from . import standard_fit
        import copy
        results = []
        for bar_trace in copy.copy(data):
            result = standard_fit.fit(bar_trace.x, bar_trace.y, fit)
            results.append(result)

            x = np.linspace(min(bar_trace.x), max(bar_trace.x), 1000)

            trace = go.Scatter(
                mode="lines",
                name=f"{fit} fit of {bar_trace.name}",
                x=x,
                y=standard_fit.get_func(fit)(x, *result[1]),
                line=dict(
                    color=bar_trace.marker.color
                )
            )

            data.append(trace)

    fig = go.Figure(
        data=(*data, *marginal_traces),
        layout=fig.layout
    )

    fig.update_layout(
        bargap=0,
        barmode=barmode
    )

    if fit is None:
        return fig
    else:
        return fig, np.array(results)
