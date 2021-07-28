""" Utilities for solving and simulating the model_analysis.

This modules contains standardized functions for solving
 and simulating throughout the analytical part of the
  project.

"""
#####################################################
# IMPORTS
#####################################################
import numpy as np
from scipy import interpolate

from src.model_analysis.solve_model import _solve_and_simulate

#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def _solve_run(program_run, controls, calibration):

    # implement program_run
    for key in program_run:
        calibration[key] = program_run[key]

    # solve and simulate
    results = _solve_and_simulate(controls, calibration)

    return results


def interpolate_1d(x_old, y_old, x_new, method="cubic"):
    return interpolate.interp1d(x_old, y_old, kind=method)(x_new)


def interpolate_2d_ordered_to_unordered(
    x_old, y_old, z_old, x_new, y_new, method="cubic"
):
    """
    Interpolate 2D data over ordered base points at unordered target points.

    :parameter:
    x_old : array [m x n]
        X-component of base points
    y_old : array [m x n]
        Y-component of base points
    z_old : array [m x n]
        Function values at base points
    x_new : array [o x p]
        X-component of target points
    y_new : array [o x p]
        Y-component of target points
    method : string
        Degree of interpolator; options
            - "linear": 1st degree spline interpolation
            - "cubic": 3rd degree spline interpolation

    :returns:
    z_new : array [o x p]
        Interpolated function values at target points

    Wrapper function applying scipy.interpolate.RectBivariateSpline.
    First column of *x_old* and first row of *y_old* are used to construct
    interpolator (implementation designed for interchangeability of methods).

    """

    # set degree of interpolator
    if method == "linear":
        kx = ky = 1
    elif method == "cubic":
        kx = ky = 3
    else:
        raise ValueError("Unknown interpolation method %r" % (method))

    # get interpolator
    interpolator = interpolate.RectBivariateSpline(
        x_old[:, 0], y_old[0, :], z_old, kx=kx, ky=ky
    )

    # call interpolation
    out = interpolator.ev(x_new, y_new)

    return out


def interpolate_2d_unordered_to_unordered(
    x_old, y_old, z_old, x_new, y_new, method="cubic"
):
    """
    Interpolate 2D data over unordered base points at unordered target points.

    :parameter:
    x_old : array [m x n]
        X-component of base points
    y_old : array [m x n]
        Y-component of base points
    z_old : array [m x n]
        Function values at base points
    x_new : array [o x p]
        X-component of target points
    y_new : array [o x p]
        Y-component of target points
    method : string
        Degree of interpolator; options
            - "linear": 1st degree spline interpolation
            - "cubic": 3rd degree spline interpolation

    :returns:
    z_new : array [o x p]
        Interpolated function values at target points

    Wrapper function applying scipy.interpolate.griddata.
    Base points, target points and function values are handled as 2d-arrays,
    although they don't need to be on a regular grid (implementation
    designed for interchangeability of methods).

    """
    shape_out = (x_new.shape[0], x_new.shape[1])

    xy_old = np.dstack((x_old, y_old)).reshape(-1, 2)
    z_old = z_old.reshape(-1, 1)
    xy_new = np.dstack((x_new, y_new)).reshape(-1, 2)

    z_new = interpolate.griddata(xy_old, z_old, xy_new, method=method)

    return z_new.reshape(shape_out)


def interpolate_2d_unordered_to_unordered_iter(
    x_old, y_old, z_old, x_new, y_new, method="cubic"
):

    out = np.full(x_new.shape, np.nan)

    for nn in range(x_old.shape[0]):
        out[nn, :] = interpolate_uex(
            y_old[nn, :], z_old[nn, :], y_new[nn, :], method=method
        )

    return out


def interpolate_extrapolate(x_old, y_old, x_new, method="cubic"):
    """
    Interpolate 1D data with *method* and extrapolate linearly.

    :parameter:
    x_old : array [m x 1]
        Base points
    y_old : array [m x 1]
        Function values at base points
    x_new : array [n x 1]
        Target points
    method : string
        Degree of interpolator; options
            - "linear": 1st degree spline interpolation
            - "cubic": 3rd degree spline interpolation

    :returns:
    y_new : array [n x 1]
        Interpolated function values at target points

    """

    f_interpolate = interpolate.interp1d(
        x_old, y_old, kind=method, bounds_error=False, fill_value=np.nan
    )
    f_extrapolate = interpolate.interp1d(
        x_old, y_old, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    out = ((x_new <= min(x_old)) + (x_new >= max(x_old))) * f_extrapolate(x_new) + (
        min(x_old) < x_new
    ) * (x_new < max(x_old)) * f_interpolate(x_new)

    return out


def interpolate_uex(x_old, y_old, x_new, method="cubic"):
    """interpolate 1d data with *method*, extrapolate linearly above grid
    and nearest below grid.
    """
    # initiate interpolants
    f_interpolate = interpolate.interp1d(
        x_old, y_old, kind=method, bounds_error=False, fill_value="extrapolate"
    )
    f_extrapolate = interpolate.interp1d(
        x_old, y_old, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    # evaluate at new points
    out = (
        (x_new <= min(x_old)) * np.full(len(x_new), y_old[0])
        + (min(x_old) < x_new)
        * (x_new < max(x_old))
        * f_interpolate(np.maximum(x_new, np.amin(x_old)))
        + (max(x_old) <= x_new) * f_extrapolate(np.maximum(x_new, np.amin(x_old)))
    )

    return out


def interpolate_uex2(x_old, y_old, x_new, method="cubic"):

    out = (
        interpolate.interp1d(
            x_old, y_old, kind=method, bounds_error=False, fill_value="extrapolate"
        )(np.maximum(x_new, min(x_old)))
        * (x_new > min(x_old))
        * (x_new < max(x_old))
        + interpolate.interp1d(
            x_old, y_old, kind="linear", bounds_error=False, fill_value="extrapolate"
        )(np.maximum(x_new, min(x_old)))
        * (x_new >= max(x_old))
        + np.full(len(y_old), y_old[0]) * (x_new <= min(x_old))
    )

    return out
