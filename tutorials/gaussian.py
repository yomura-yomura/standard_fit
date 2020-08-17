#!/usr/bin/env python3
import numpy as np
import standard_fit as sf


if __name__ == "__main__":
    x = np.random.normal(3, 2, size=100)
    sf.gaussian_fit_and_show(x)
