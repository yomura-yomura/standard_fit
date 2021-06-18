#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly_utility.express as pux
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.random.normal(size=10_000)
    col = np.random.choice(np.arange(3), size=x.size)
    row = np.random.choice(np.arange(3), size=x.size)

    fig = sfpx.fit(
        pux.histogram(x=x, facet_col=col, facet_row=row),
        fit_type="gaussian",
        row="all", col="all",
        # annotation_kwargs=dict(inside=False)
    )
    fig.show()

