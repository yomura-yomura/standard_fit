

__all__ = []


def get_func(fit_type: str):
    if fit_type not in __all__:
        raise ValueError(f"{fit_type} not defined.")
    return globals()[fit_type]
