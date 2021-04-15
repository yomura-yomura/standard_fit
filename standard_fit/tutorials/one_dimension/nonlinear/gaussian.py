#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.random.normal(3, 2, size=1_000_000)
    n_nan = int(np.random.normal(x.size * 0.01))
    n_inf = int(np.random.normal(x.size * 0.01))
    x[np.random.choice(x.size, size=n_nan, replace=False)] = np.nan
    x[np.random.choice(x.size, size=n_inf // 2, replace=False)] = np.inf
    x[np.random.choice(x.size, size=n_inf // 2, replace=False)] = -np.inf

    fig = sfpx.histogram(x=x, fit_type="gaussian", fit_stats=True, marginal="box")
    fig.show()
