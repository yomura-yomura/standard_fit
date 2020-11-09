import jax
import jax.numpy as jnp


__all__ = [
    "gaussian", "power_law", "exp", "log",
    "sin",
    "exp10",
    "tanh",
    # "approx_landau", "kde",
    *[f"pol{i}" for i in range(4)]
]


def gaussian(x, A, μ, σ):
    return A * jnp.exp(-0.5 * ((x - μ) / σ) ** 2)


def power_law(x, A, s):
    return A * jnp.power(x, s)


def exp(x, p0, p1):
    return p0 * jnp.exp(p1 * x)


def log(x, p0, p1):
    return p0 * jnp.log(p1 * x)


def sin(x, A, ω, x0):
    return A * jnp.sin(ω * (x - x0))


def exp10(x, p0, p1):
    return p0 * jnp.power(10, p1 * x)


def tanh(x, C, a, x0, y0):
    return C * jnp.tanh(a * (x - x0)) + y0


def _make_pol_n(n):
    kwargs = [f"p{i}" for i in range(n + 1)]
    formula = "+".join([f"{p}*x**{i}" for i, p in enumerate(kwargs)])
    unpacked_kwargs = ", ".join(kwargs)
    exec(f"def pol{n}(x, {unpacked_kwargs}): return {formula}")
    func = eval(f"pol{n}")
    globals()[f"pol{n}"] = func
    return func


for i in range(4):
    _make_pol_n(i)


for fn in __all__:
    locals()[fn] = jax.jit(jax.grad(locals()[fn]))


def _validate_fit_type(fit_type):
    if fit_type not in __all__:
        raise ValueError(f"{fit_type} not defined in {__all__}")


def get_func(fit_type):
    fit_type = fit_type.replace(" ", "_")
    _validate_fit_type(fit_type)
    return globals()[fit_type]
