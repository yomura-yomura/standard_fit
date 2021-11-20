import standard_fit as sf
import standard_fit.plotly.express as sfpx
import numpy as np
import plotly



if __name__ == "__main__":
    def custom_func(x, a1, a2):
        return a1 * np.sin(x) + a2 * np.cos(x/2)

    sf.add_function("custom func", custom_func)

    x = np.linspace(0, 2*np.pi, 100)
    y = 20 * np.sin(x) + 0.5 * np.cos(x/2)

    sfpx.scatter(x=x, y=y, fit_type="custom func").show()
