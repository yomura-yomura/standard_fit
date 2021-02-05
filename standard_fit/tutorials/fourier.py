#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly
import plotly.express as px
import standard_fit as sf
import scipy.stats
plotly.io.renderers.default = "browser"


if __name__ == "__main__":
    x = np.linspace(0, 6, 1000)
    y = 10 + sum(k * np.sin(k * x) for k in [1, 5, 9, 30]) + np.random.normal(0, 2, size=len(x))
    fig = sfpx.fit(
        px.scatter(x=x, y=y, error_y=[2] * len(x)),
        "fourier30",
        annotation_kwargs=dict(inside=False),
        # fit_kwargs=dict(lasso_lambda="auto")
    )
    fig.show(config={"editable": True})

    fr = fig._fit_results.flatten()[0]
    t = fr["params"] / fr["err_params"] ** 2
    p = scipy.stats.t.sf(t, fr["ndf"] - 1)
    indices = np.where(p == 0)[0]
    print(np.take(sf.regression.get_parameter_names("fourier30"), indices))
    print(np.take(fr["params"], indices))
