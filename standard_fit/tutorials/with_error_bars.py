#!/usr/bin/env python3
import numpy as np
import numpy_utility as npu
import standard_fit.plotly.express as sfpx
import plotly_utility.express as pux
import plotly.graph_objs as go
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    np.random.seed(0)
    size = 100_000
    x = np.random.exponential(size=size)
    y = x*x
    y += np.random.normal(scale=y*y)

    bins, mean, sigma = npu.omr.binning(x, y, allowed_minimum_size=10)

    fig = go.Figure(
        data=[
            go.Scattergl(
                name="original",
                mode="markers",
                marker_size=1.5,
                x=x,
                y=y
            ),
            go.Scattergl(
                name="averaged with binning",
                mode="markers",
                x=bins,
                y=mean,
                # error_x=dict(
                #     array=[1] * len(sigma)
                # ),
                error_y=dict(
                    array=sigma
                )
            )
        ]
    )

    fig = sfpx.fit(fig, "pol2", row=None, col=None, i_data=2)

    fig.update_yaxes(range=[-100, 300])

    fig.show()

    # px.scatter(x=bins, y=mean, error_y=sigma).show()

