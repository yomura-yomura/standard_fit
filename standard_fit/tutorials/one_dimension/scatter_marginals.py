#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.random.normal(size=10_000)
    y = np.random.normal(x, size=len(x))

    sfpx.scatter(
        x=x, y=y, fit_type="pol1",
        fit_marginal_x="gaussian", fit_marginal_y="gaussian",
        annotation_kwargs=dict(use_font_size=True, font_size=20)
    ).show()
