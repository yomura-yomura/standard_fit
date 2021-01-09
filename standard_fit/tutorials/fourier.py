#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express as sfpx
import plotly
plotly.io.renderers.default = "browser"
import standard_fit as sf


if __name__ == "__main__":
    x = np.linspace(0, 10, 1000)
    y = 10 + sum(k * np.sin(k * x) for k in [1, 5, 9, 30]) + np.random.normal(0, 2, size=len(x))
    fig = sfpx.scatter(x, y, "fourier30", error_y=[2] * len(x),
                       # fit_kwargs=dict(fit_kwargs=dict(lasso_lambda="auto"))
                       )
    fig.show(config={"editable": True})

    fr = fig._fit_results.flatten()[0]
    t = fr["params"] / fr["err_params"] ** 2
    import scipy.stats
    p = scipy.stats.t.sf(t, fr["ndf"] - 1)
    indices = np.where(p == 0)[0]
    print(np.take(sf.regression.get_parameter_names("fourier30"), indices))
    print(np.take(fr["params"], indices))
