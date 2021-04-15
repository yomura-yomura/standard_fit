

def estimate_initial_guess(fit_type, x, y):
    if fit_type in globals():
        return globals()[fit_type](x, y)
    else:
        raise NotImplementedError(fit_type)
