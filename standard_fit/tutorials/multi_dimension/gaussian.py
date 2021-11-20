#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import scipy.stats
import plotly



if __name__ == "__main__":
    np.random.seed(0)
    xv, yv = np.meshgrid(np.linspace(0, 20, 30), np.linspace(-20, 0, 30))
    x = np.stack((xv.flatten(), yv.flatten()), axis=-1)

    mean = np.array([10, -10])
    # cov = 5 * np.identity(2)
    cov = [
        [10, 5],
        [5, 10]
    ]
    error_y = [0.001] * len(x)
    y = scipy.stats.multivariate_normal.pdf(x, mean, cov) + np.random.normal(0, error_y)

    fig = sfpx.scatter_3d(
        x=x[:, 0], y=x[:, 1], z=y, error_z=error_y, fit_type="gaussian2d",
        annotation_kwargs=dict(display_matrix=True)
    )
    fig.show()
