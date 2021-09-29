import numpy as np
import plotly.express as px
import standard_fit.plotly.express as sfpx
import scipy.stats


print("y = -2x + 1")
np.random.seed(1000)

x = np.linspace(-10, 10, 150)
y_sigma = [2] * len(x)
y = -2 * x + 1 + np.random.normal(0, y_sigma)
fig = px.scatter(x=x, y=y, error_y=y_sigma)
sfpx.fit(fig, "pol1")
fig.write_image("pol1.png", width=1200, height=750)


print("y = 2sin(x) + 5cos(2x) + 9cos(5x)")
np.random.seed(1000)

x = np.linspace(-10, 10, 1000)
y_sigma = [2] * len(x)
y = 2 * np.sin(x) + 5 * np.cos(2 * x) + 9 * np.cos(5 * x) + np.random.normal(0, y_sigma)

fig = px.scatter(title="Fourier-series fitting", x=x, y=y, error_y=y_sigma)
sfpx.fit(fig, "fourier5", annotation_kwargs=dict(inside=False))
fig.write_image("fs.png", width=1200, height=750)

fig = px.scatter(title="Fourier-series fitting with LASSO regularization (λ=0.1)", x=x, y=y, error_y=y_sigma)
sfpx.fit(fig, "fourier5", annotation_kwargs=dict(inside=False),
         fit_kwargs=dict(linear_regression_kwargs=dict(lasso_lambda=0.1)))
fig.write_image("fs_lasso.png", width=1200, height=750)


print("Gaussian x ~ N(5, 2)")
np.random.seed(1000)

x = np.random.normal(5, 2, size=1_000_000)
fig = sfpx.histogram(x=x, fit_type="gaussian")
fig.write_image("gaus.png", width=1200, height=750)


print("Gaussian x ~ N(5, 2) (Unbinned Maximum Likelihood fit)")
np.random.seed(1000)

x = np.random.normal(5, 2, size=1_000)
fig = sfpx.histogram(x=x, fit_type="gaussian", umlf=True, histnorm="probability density")
fig.write_image("gaus_umlf.png", width=1200, height=750)


print("2D Gaussian x ~ N([10, -10], [[10, 5], [5, 10]])")
np.random.seed(1000)

xv, yv = np.meshgrid(np.linspace(0, 20, 30), np.linspace(-20, 0, 30))
x = np.stack((xv.flatten(), yv.flatten()), axis=-1)

mean = [10, -10]
cov = [
    [10, 5],
    [5, 10]
]
error_y = [0.001] * len(x)

y = scipy.stats.multivariate_normal.pdf(x, mean, cov) + np.random.normal(0, error_y)

fig = sfpx.scatter_3d(
    x=x[:, 0], y=x[:, 1], z=y, error_z=error_y, fit_type="gaussian2d",
    annotation_kwargs=dict(display_matrix=True)
)
fig.write_image("gaus2d.png", width=1200, height=750)
