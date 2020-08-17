import sys
import numpy as np

sys.path.append("..")
import standard_fit as sf


if __name__ == "__main__":
    pol3 = False
    gaus = True

    if pol3:
        x = np.linspace(-5, 5, 20)
        y = np.poly1d([1, 3, -9, 4])(x) + np.random.normal(0, 2, size=len(x))
        sf.fit_and_show(x, y, "pol3")
    if gaus:
        x = np.random.normal(3, 2, size=100)

        import standard_fit.unbinned_maximum_likelihood_fit as umlf
        umlf.fit_and_show(x, "gaussian")

        sf.gaussian_fit_and_show(x)



