#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.linspace(-5, 5, 2000)
    y = np.poly1d([1, 3, -9, 4])(x) + np.random.normal(0, 2, size=len(x))
    fig = sfpx.scatter(x, y, "pol3", error_y=[2] * len(x))
    fig.show(config={"editable": True})
