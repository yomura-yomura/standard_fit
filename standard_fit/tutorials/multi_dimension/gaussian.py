#!/usr/bin/env python3
import numpy as np
import standard_fit as sf
import standard_fit.plotly.express as sfpx
import plotly
import scipy.stats
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    np.random.seed(0)
    x = np.stack([e.flatten() for e in np.meshgrid(np.linspace(0, 20, 30), np.linspace(-20, 0, 30))], axis=-1)

    mean = np.array([10, -10])
    cov = 5 * np.identity(2)
    # cov = [
    #     [5, 2],
    #     [2, 10]
    # ]
    y_sigma = 0.001

    y = scipy.stats.multivariate_normal.pdf(x, mean, cov) + np.random.normal(0, y_sigma, size=len(x))

    fig = sfpx.scatter_3d(
        x=x[:, 0], y=x[:, 1], z=y, error_z=[y_sigma] * len(y), fit_type="gaussian2d"
    )
    fig.show()
