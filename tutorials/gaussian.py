#!/usr/bin/env python3
import numpy as np
import standard_fit.plotly.express


if __name__ == "__main__":
    x = np.random.normal(3, 2, size=1_000_000)
    standard_fit.plotly.express.histogram(x, do_fit=True, fit_stats=True).show()
