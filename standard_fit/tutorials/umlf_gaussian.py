#!/usr/bin/env python3
import numpy as np
import standard_fit as sf
import standard_fit.unbinned_maximum_likelihood_fit as umlf


if __name__ == "__main__":
    x = np.random.normal(3, 2, size=100)
    umlf.fit_and_show(x, "gaussian")
    sf.gaussian_fit_and_show(x)

