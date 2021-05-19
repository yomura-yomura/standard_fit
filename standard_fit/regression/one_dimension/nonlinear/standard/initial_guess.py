# Estimate initial guess
import numpy as np
import numpy_utility as npu


__all__ = ["estimate_initial_guess"]


def na_pol1(x, y):
    p1 = (max(y) - min(y)) / (max(x) - min(x))
    p0 = min(y) - min(x) * p1
    return p0, p1


def gaussian(x, y):
    mean = x[np.argmax(y)]
    try:
        std = np.average(np.abs(x - mean), weights=y)
    except ZeroDivisionError:
        std = 0
    a = np.max(y) * np.sqrt(2 * np.pi) * std
    return a, mean, std


def power_law(x, y):
    min_i = np.argmin(x)
    max_i = np.argmax(x)
    s = (np.log(y[max_i]) - np.log(y[min_i])) / (np.log(x[max_i]) - np.log(x[min_i]))
    a = y[min_i] * np.power(x[min_i], -s)
    return a, s


def exp(x, y):
    min_i = np.argmin(x)
    max_i = np.argmax(x)
    p1 = (np.log(y[max_i]) - np.log(y[min_i])) / (x[max_i] - x[min_i])
    med_i = np.searchsorted(x, np.mean(x), sorter=np.argsort(x))
    p0 = y[med_i] * np.exp(-p1 * x[med_i])
    return p0, p1


def log(x, y):
    min_i = np.argmin(x)
    max_i = np.argmax(x)
    p1 = (np.log(x[max_i]) - np.log(x[min_i])) / (y[max_i] - y[min_i])
    p0 = x[min_i] * np.exp(-p1 * y[min_i])
    return p0, p1


def sin(x, y):
    assert np.allclose((x[1:] - x[:-1])[1:], (x[1:] - x[:-1])[:-1])
    freq_on_indices = abs(np.fft.fftfreq(y.size)[np.abs(np.fft.fft(y))[1:].argmax() + 1])
    lambda_on_indices = int(1 / freq_on_indices)
    reshaped_y = npu.reshape(y, (-1, lambda_on_indices), drop=True)

    phi_on_indices = (
         (reshaped_y.argmax(axis=1).mean() - lambda_on_indices / 4) +
         (reshaped_y.argmin(axis=1).mean() - lambda_on_indices * 3 / 4)
     ) / 2

    A = (reshaped_y.max(axis=1).mean() - reshaped_y.min(axis=1).mean()) / 2

    scale = x[1] - x[0]
    return A, 2 * np.pi * freq_on_indices / scale, phi_on_indices * scale


def exp10(x, y):
    min_i = np.argmin(x)
    max_i = np.argmax(x)
    p1 = (np.log10(y[max_i]) - np.log10(y[min_i])) / (x[max_i] - x[min_i])
    med_i = np.searchsorted(x, np.mean(x))
    p0 = y[med_i] * np.power(10, -p1 * x[med_i])
    return p0, p1


def tanh(x, y):
    C = (np.max(y) - np.min(y)) * 0.5
    y0 = (np.min(y) + np.max(y)) * 0.5
    x0 = x[np.searchsorted(y, y0)]
    a = 1
    return C, a, x0, y0


def approx_landau(x, y):
    max_i = np.argmax(y)
    A = y[max_i]
    m = x[max_i]
    s = A * 1
    return A, m, s


def estimate_initial_guess(fit_type: str, x, y):
    if fit_type in (k for k in globals() if not k.startswith("_")):
        return globals()[fit_type](x, y)
    else:
        raise NotImplementedError(fit_type)


def log_gaussian(x, y):
    return gaussian(np.log10(x), y)


def log_gaussian2(x, y):
    A, mu, sigma = gaussian(np.log10(x), y)
    sigma2 = sigma ** 2
    return A, np.exp(mu + sigma2 / 2), np.sqrt(np.exp(2 * mu + sigma2) * (np.exp(sigma2) - 1))
