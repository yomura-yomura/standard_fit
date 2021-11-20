#!/usr/bin/env python3
"""
Test
"""
import numpy as np
import standard_fit as sf
import standard_fit.plotly.express as sfpx
import plotly



if __name__ == "__main__":
    def td_func(x, A, B):
        R, rho = np.rollaxis(x, axis=1)
        td = 2.6 * (1 + R / 30) ** A * rho ** B
        return td

    sf.add_function("td", td_func, is_multivariate=True)

    import pandas as pd
    df = pd.read_csv(
        "newtd.txt", delim_whitespace=True, skiprows=1, header=None,
        names=["R", "rho", "td[micro]",  "zenith", "log8"]
    )

    fig = sfpx.scatter_3d(df, x="R", y="rho", z="td[micro]", fit_type="td")
    fig.show()
