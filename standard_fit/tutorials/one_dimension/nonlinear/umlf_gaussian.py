#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly



if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.normal(3, 2, size=100_000)

    print("\n* Unbinned maximum likelihood fit")
    sfpx.histogram(
        x=x, fit_type="gaussian", umlf=True, histnorm="probability density",
        title="Unbinned maximum likelihood fit"
    ).update_xaxes(range=(-5, 11)).show()

    print("\n* standard fit")
    sfpx.histogram(
        x=x, fit_type="gaussian", histnorm="probability density", title="Gaussian fit"
    ).update_xaxes(range=(-5, 11)).show()

