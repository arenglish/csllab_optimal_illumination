from scipy import interpolate as inter


def interpolate(y, x, x_interp):
    f = inter.interp1d(x, y, bounds_error=False, fill_value=0)
    ynew = f(x_interp)
    return ynew
