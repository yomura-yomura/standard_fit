#!/usr/bin/env python3
import numpy as np
import standard_fit as sf


if __name__ == "__main__":
    x = np.linspace(-5, 5, 20)
    y = np.poly1d([1, 3, -9, 4])(x) + np.random.normal(0, 2, size=len(x))
    sf.fit_and_show(x, y, "pol3")
