#!/usr/bin/env python3
import numpy as np
import standard_fit as sf
import standard_fit.plotly.express as sfpx
import plotly
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.normal(3, 2, size=100_000)
    sfpx.histogram(
        x, fit_type="gaussian", umlf=True, histnorm="probability density",
        title="Unbinned maximum likelihood fit"
    ).update_xaxes(range=(-5, 11)).show()
    sfpx.histogram(
        x, fit_type="gaussian", histnorm="probability density", title="Gaussian fit"
    ).update_xaxes(range=(-5, 11)).show()
