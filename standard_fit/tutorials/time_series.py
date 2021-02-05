# #!/usr/bin/env python3
# import numpy as np
# import standard_fit.plotly.express as sfpx
# import plotly
# plotly.io.renderers.default = "browser"
#
#
# if __name__ == "__main__":
#     x = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-12-31"), np.timedelta64(1, "h"))
#     y = 10 * np.sin((x - x.min()).astype(int) / (7*24) * 2*np.pi) + np.random.normal(0, 3, size=x.size)
#
#     fig = sfpx.scatter(x, y, fit_type="sin", fit_kwargs=dict(datetime_type="ns"))
#     fig.show()

