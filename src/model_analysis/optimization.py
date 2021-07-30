"""
Find optimal life cycle profiles of UI benefit replacement rates and tax rates.
"""
#####################################################
# IMPORTS
#####################################################
import copy
import json
# import src.utilities.istarmap_3_7  # noqa, noreorder

import src.utilities.istarmap_3_8  # noqa, noreorder
import multiprocessing
import sys
import warnings

import numpy as np
import tqdm
from scipy import interpolate

from bld.project_paths import project_paths_join as ppj
from src.model_analysis.run_utils import _solve_run
from src.utilities.optimization_utils import get_step_size


#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def _eval_rate(coefficients, controls, calibration):
    """
    Wrapper function for minimization over [ui_rate].

    :parameter:
    coefficients : array
        Value of [ui_rate] at which to solve the model_analysis.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in
        *solve_model*)

    :returns:
    objective : float
        Value of objective function at *coefficients*
    equilibrium_instrument_rate : float
        Value of instrument rate that ensures balanced budget at *coefficients*

    """

    # update calibration
    calibration["ui_replacement_rate_vector"] = np.full(
        (n_types, n_periods_working), coefficients[0]
    )

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    objective = results["average_pv_utility_computed_corrected"]
    equilibrium_quantities = {
        "instrument": results["equilibrium_instrument_rate"],
        "transfers_pensions": results["equilibrium_transfers_pensions"],
        "transfers_lumpsum": results["equilibrium_transfers_lumpsum"],
    }

    return objective, equilibrium_quantities


def _eval_rate_floor_cap(coefficients, controls, calibration):
    """
    Wrapper function for minimization over [ui_rate, ui_floor, ui_cap].

    :parameter:
    coefficients : array
        Values of [ui_rate, ui_floor, ui_cap] at which to solve the model_analysis.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in
        *solve_model*)

    :returns:
    objective : float
        Value of objective function at *coefficients*
    equilibrium_instrument_rate : float
        Value of instrument rate that ensures balanced budget at *coefficients*

    """

    # update calibration
    calibration["ui_replacement_rate_vector"] = np.full(
        (n_types, n_periods_working), coefficients[0]
    )
    calibration["ui_floor"] = coefficients[1]
    calibration["ui_cap"] = coefficients[2]

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    objective = results["average_pv_utility_computed_corrected"]
    equilibrium_quantities = {
        "instrument": results["equilibrium_instrument_rate"],
        "transfers_pensions": results["equilibrium_transfers_pensions"],
        "transfers_lumpsum": results["equilibrium_transfers_lumpsum"],
    }

    return objective, equilibrium_quantities


def _eval_rate_type(coefficients, controls, calibration):
    """
    Wrapper function for minimization over ui replacement rate vector defined
    by *coefficients*.

    :parameter:
    coefficients : array
        UI replacement rates at UI spline nodes for which to solve the model_analysis.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in
        *solve_model*)

    :returns:
    objective : float
        Value of objective function at *coefficients*
    equilibrium_instrument_rate : float
        Value of instrument rate that ensures balanced budget at *coefficients*

    """

    # load calibration
    n_periods_working = calibration["n_periods_working"]

    # get UI replacement rate vector
    ui_replacement_rate_vector = np.repeat(coefficients, n_periods_working).reshape(
        (n_types, n_periods_working)
    )

    # update calibration
    calibration["ui_replacement_rate_vector"] = ui_replacement_rate_vector.tolist()

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    objective = results["average_pv_utility_computed_corrected"]
    equilibrium_quantities = {
        "instrument": results["equilibrium_instrument_rate"],
        "transfers_pensions": results["equilibrium_transfers_pensions"],
        "transfers_lumpsum": results["equilibrium_transfers_lumpsum"],
    }

    return objective, equilibrium_quantities


def _eval_rate_vector(coefficients, controls, calibration):
    """
    Wrapper function for minimization over ui replacement rate vector defined
    by *coefficients*.

    :parameter:
    coefficients : array
        UI replacement rates at UI spline nodes for which to solve the model_analysis.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in
        *solve_model*)

    :returns:
    objective : float
        Value of objective function at *coefficients*
    equilibrium_instrument_rate : float
        Value of instrument rate that ensures balanced budget at *coefficients*

    """

    # load calibration
    n_periods_working = calibration["n_periods_working"]
    ui_replacement_rate_grid_reduced = np.array(calibration["ui_grid"])
    ui_replacement_rate_min = calibration["ui_replacement_rate_min"]

    # get UI replacement rate vector
    ui_replacement_rate_vector = interpolate.PchipInterpolator(
        ui_replacement_rate_grid_reduced, coefficients
    )(np.linspace(0, n_periods_working - 1, n_periods_working))
    ui_replacement_rate_vector = np.maximum(
        ui_replacement_rate_vector, ui_replacement_rate_min
    )
    ui_replacement_rate_vector = np.tile(ui_replacement_rate_vector, n_types).reshape(
        (n_types, n_periods_working)
    )

    # update calibration
    calibration["ui_replacement_rate_vector"] = ui_replacement_rate_vector.tolist()

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    objective = results["average_pv_utility_computed_corrected"]
    equilibrium_quantities = {
        "instrument": results["equilibrium_instrument_rate"],
        "transfers_pensions": results["equilibrium_transfers_pensions"],
        "transfers_lumpsum": results["equilibrium_transfers_lumpsum"],
    }

    return objective, equilibrium_quantities


def _eval_rate_age_type(coefficients, controls, calibration):
    """
    Wrapper function for minimization over ui replacement rate vector defined
    by *coefficients*.

    :parameter:
    coefficients : array
        UI replacement rates at UI spline nodes for which to solve the model_analysis.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in
        *solve_model*)

    :returns:
    objective : float
        Value of objective function at *coefficients*
    equilibrium_instrument_rate : float
        Value of instrument rate that ensures balanced budget at *coefficients*

    """

    # load calibration
    n_periods_working = calibration["n_periods_working"]
    ui_grid_reduced = np.array(calibration["ui_grid"])
    ui_replacement_rate_min = calibration["ui_replacement_rate_min"]

    # compute derived variables
    ui_replacement_rate_grid = np.linspace(0, n_periods_working - 1, n_periods_working)
    ui_replacement_rate_grid_reduced_size = len(ui_grid_reduced)

    # get UI replacement rate vector
    ui_replacement_rate_vector = np.full((n_types, n_periods_working), np.nan)
    for type_idx in range(n_types):
        idx_start = ui_replacement_rate_grid_reduced_size * type_idx
        idx_end = ui_replacement_rate_grid_reduced_size * (type_idx + 1)
        ui_replacement_rate_vector[type_idx, :] = interpolate.PchipInterpolator(
            ui_grid_reduced, coefficients[idx_start:idx_end]
        )(ui_replacement_rate_grid)
    ui_replacement_rate_vector = np.maximum(
        ui_replacement_rate_vector, ui_replacement_rate_min
    )

    # update calibration
    calibration["ui_replacement_rate_vector"] = ui_replacement_rate_vector.tolist()

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    objective = results["average_pv_utility_computed_corrected"]
    equilibrium_quantities = {
        "instrument": results["equilibrium_instrument_rate"],
        "transfers_pensions": results["equilibrium_transfers_pensions"],
        "transfers_lumpsum": results["equilibrium_transfers_lumpsum"],
    }

    return objective, equilibrium_quantities


def _jacobian_rate(coefficients, controls, calibration):
    """
    Wrapper function to compute two-sided gradient of objective function
    w.r.t. [ui_rate] using finite differences.

    :parameter:
    coefficients : array
        Value of [ui_rate] at which to compute the gradient.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in *solve_model*)

    :returns:
    jacobian : array
        Gradient of objective function at point described by *coefficients*

    _JACOBIAN calculates ...  # todo: complete docstring

    """

    # load controls
    show_progress = controls["show_progress"]
    n_workers = controls["n_workers"]
    step_size_init = controls["step_size_jacobian"]

    # calculate control variables
    n_coefficients = coefficients.shape[0]
    n_runs = n_coefficients * 2

    # prepare computation of Jacobian
    step_size_diff = step_size_init * np.maximum(abs(coefficients), 1)
    delta = np.full(n_coefficients, np.nan)
    fx = np.full(n_runs, np.nan)

    coefficients_all = np.repeat(coefficients, n_runs).reshape(-1, n_runs)
    for idx in range(n_coefficients):
        coefficients_all[idx, idx] += step_size_diff[idx]
        coefficients_all[idx, idx + n_coefficients] += -step_size_diff[idx]
        delta[idx] = (
            coefficients_all[idx, idx] - coefficients_all[idx, idx + n_coefficients]
        )

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": np.full(
                        (n_types, n_periods_working), coefficients_all[0, run_idx]
                    ),
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Jacobian",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(n_runs):
        fx[run_idx] = np.squeeze(out[run_idx]["average_pv_utility_computed_corrected"])

    # reshape
    fx = np.moveaxis(np.stack((fx[:n_coefficients], fx[n_coefficients:])), 0, -1)

    jacobian = np.full(n_coefficients, np.nan)
    for idx in range(n_coefficients):
        jacobian[idx] = (fx[idx, 0] - fx[idx, 1]) / delta[idx]

    return jacobian


def _jacobian_rate_floor_cap(coefficients, controls, calibration):
    """
    Wrapper function to compute two-sided gradient of objective function
    w.r.t. the vector [ui_rate, ui_floor, ui_cap] using finite differences.

    :parameter:
    coefficients : array
        Values of [ui_rate, ui_floor, ui_cap] at which to compute the gradient.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in *solve_model*)

    :returns:
    jacobian : array
        Gradient of objective function at point described by *coefficients*

    _JACOBIAN calculates ...  # todo: complete docstring

    """

    # load controls
    show_progress = controls["show_progress"]
    n_workers = controls["n_workers"]
    step_size_init = controls["step_size_jacobian"]

    # calculate control variables
    n_coefficients = coefficients.shape[0]
    n_runs = n_coefficients * 2

    # prepare computation of Jacobian
    step_size_diff = step_size_init * np.maximum(abs(coefficients), 1)
    delta = np.full(n_coefficients, np.nan)
    fx = np.full(n_runs, np.nan)

    coefficients_all = np.repeat(coefficients, n_runs).reshape(-1, n_runs)
    for idx in range(n_coefficients):
        coefficients_all[idx, idx] += step_size_diff[idx]
        coefficients_all[idx, idx + n_coefficients] += -step_size_diff[idx]
        delta[idx] = (
            coefficients_all[idx, idx] - coefficients_all[idx, idx + n_coefficients]
        )

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": np.full(
                        (n_types, n_periods_working), coefficients_all[0, run_idx]
                    ),
                    "ui_floor": coefficients_all[1, run_idx],
                    "ui_cap": coefficients_all[2, run_idx],
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Jacobian",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(n_runs):
        fx[run_idx] = np.squeeze(out[run_idx]["average_pv_utility_computed_corrected"])

    # reshape
    fx = np.moveaxis(np.stack((fx[:n_coefficients], fx[n_coefficients:])), 0, -1)

    jacobian = np.full(n_coefficients, np.nan)
    for idx in range(n_coefficients):
        jacobian[idx] = (fx[idx, 0] - fx[idx, 1]) / delta[idx]

    return jacobian


def _jacobian_rate_type(coefficients, controls, calibration):
    """
    Compute two-sided gradient of a expected average value at model_analysis entry w.r.t. the
    parameters of the unemployment insurance rate using finite differences.

    :parameter:
    coefficients : array
        Coordinates at which to compute gradient.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in *solve_model*)

    :returns:
    jacobian : array
        Gradient of objective function at point described by *coefficients*

    _JACOBIAN calculates ...  # todo: complete docstring

    """

    # load controls
    show_progress = controls["show_progress"]
    n_workers = controls["n_workers"]
    step_size_init = controls["step_size_jacobian"]

    # load calibration
    n_periods_working = calibration["n_periods_working"]

    # calculate control variables
    n_coefficients = coefficients.shape[0]
    n_runs = n_coefficients * 2

    # prepare computation of Jacobian
    step_size_diff = step_size_init * np.maximum(abs(coefficients), 1)
    delta = np.full(n_coefficients, np.nan)
    fx = np.full(n_runs, np.nan)

    coefficients_all = np.repeat(coefficients, n_runs).reshape(-1, n_runs)
    for idx in range(n_coefficients):
        coefficients_all[idx, idx] += step_size_diff[idx]
        coefficients_all[idx, idx + n_coefficients] += -step_size_diff[idx]
        delta[idx] = (
            coefficients_all[idx, idx] - coefficients_all[idx, idx + n_coefficients]
        )

    ui_replacement_rate_vector_all = np.full(
        (n_types, n_periods_working, n_runs), np.nan
    )
    for run_idx in range(n_runs):
        ui_replacement_rate_vector_tmp = np.repeat(
            coefficients_all[:, run_idx], n_periods_working
        ).reshape((n_types, n_periods_working))
        ui_replacement_rate_vector_all[:, :, run_idx] = ui_replacement_rate_vector_tmp

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": ui_replacement_rate_vector_all[
                        :, :, run_idx
                    ]
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Jacobian",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(n_runs):
        fx[run_idx] = np.squeeze(out[run_idx]["average_pv_utility_computed_corrected"])

    # reshape
    fx = np.moveaxis(np.stack((fx[:n_coefficients], fx[n_coefficients:])), 0, -1)

    jacobian = np.full(n_coefficients, np.nan)
    for idx in range(n_coefficients):
        jacobian[idx] = (fx[idx, 0] - fx[idx, 1]) / delta[idx]

    return jacobian


def _jacobian_rate_vector(coefficients, controls, calibration):
    """
    Compute two-sided gradient of a expected average value at model_analysis entry w.r.t. the
    parameters of the unemployment insurance rate using finite differences.

    :parameter:
    coefficients : array
        Coordinates at which to compute gradient.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in *solve_model*)

    :returns:
    jacobian : array
        Gradient of objective function at point described by *coefficients*

    _JACOBIAN calculates ...  # todo: complete docstring

    """

    # load controls
    show_progress = controls["show_progress"]
    n_workers = controls["n_workers"]
    step_size_init = controls["step_size_jacobian"]

    # load calibration
    n_periods_working = calibration["n_periods_working"]
    ui_replacement_rate_grid_reduced = np.array(calibration["ui_grid"])
    ui_replacement_rate_min = calibration["ui_replacement_rate_min"]

    # calculate control variables
    n_coefficients = coefficients.shape[0]
    n_runs = n_coefficients * 2

    # prepare computation of Jacobian
    step_size_diff = step_size_init * np.maximum(abs(coefficients), 1)
    delta = np.full(n_coefficients, np.nan)
    fx = np.full(n_runs, np.nan)

    coefficients_all = np.repeat(coefficients, n_runs).reshape(-1, n_runs)
    for idx in range(n_coefficients):
        coefficients_all[idx, idx] += step_size_diff[idx]
        coefficients_all[idx, idx + n_coefficients] += -step_size_diff[idx]
        delta[idx] = (
            coefficients_all[idx, idx] - coefficients_all[idx, idx + n_coefficients]
        )

    ui_replacement_rate_vector_all = np.full(
        (n_types, n_periods_working, n_runs), np.nan
    )
    for run_idx in range(n_runs):
        ui_replacement_rate_vector_tmp = interpolate.PchipInterpolator(
            ui_replacement_rate_grid_reduced, coefficients_all[:, run_idx]
        )(np.linspace(0, n_periods_working - 1, n_periods_working))
        ui_replacement_rate_vector_tmp = np.maximum(
            ui_replacement_rate_vector_tmp, ui_replacement_rate_min
        )
        ui_replacement_rate_vector_tmp = np.tile(
            ui_replacement_rate_vector_tmp, n_types
        ).reshape((n_types, n_periods_working))
        ui_replacement_rate_vector_all[:, :, run_idx] = ui_replacement_rate_vector_tmp

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": ui_replacement_rate_vector_all[
                        :, :, run_idx
                    ]
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Jacobian",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(n_runs):
        fx[run_idx] = np.squeeze(out[run_idx]["average_pv_utility_computed_corrected"])

    # reshape
    fx = np.moveaxis(np.stack((fx[:n_coefficients], fx[n_coefficients:])), 0, -1)

    jacobian = np.full(n_coefficients, np.nan)
    for idx in range(n_coefficients):
        jacobian[idx] = (fx[idx, 0] - fx[idx, 1]) / delta[idx]

    return jacobian


def _jacobian_rate_age_type(coefficients, controls, calibration):
    """
    Compute two-sided gradient of a expected average value at model_analysis entry w.r.t. the
    parameters of the unemployment insurance rate using finite differences.

    :parameter:
    coefficients : array
        Coordinates at which to compute gradient.
    controls : dict
        Collection of control variables for computation (details see
        description of *qnewton*)
    calibration : dict
        Collection of model_analysis parameters (details see description in *solve_model*)

    :returns:
    jacobian : array
        Gradient of objective function at point described by *coefficients*

    _JACOBIAN calculates ...  # todo: complete docstring

    """

    # load controls
    show_progress = controls["show_progress"]
    n_workers = controls["n_workers"]
    step_size_init = controls["step_size_jacobian"]

    # load calibration
    n_periods_working = calibration["n_periods_working"]
    ui_replacement_rate_grid_reduced = np.array(calibration["ui_grid"])
    ui_replacement_rate_min = calibration["ui_replacement_rate_min"]

    # compute derived variables
    ui_replacement_rate_grid = np.linspace(0, n_periods_working - 1, n_periods_working)
    ui_replacement_rate_grid_reduced_size = len(ui_replacement_rate_grid_reduced)

    # calculate control variables
    n_coefficients = len(coefficients)
    n_runs = n_coefficients * 2

    # prepare computation of Jacobian
    step_size_diff = step_size_init * np.maximum(abs(coefficients), 1)
    delta = np.full(n_coefficients, np.nan)
    fx = np.full((n_coefficients, 2), np.nan)

    coefficients_all = np.repeat(coefficients, n_runs).reshape((n_coefficients, n_runs))
    for idx in range(n_coefficients):
        coefficients_all[idx, idx] += step_size_diff[idx]
        coefficients_all[idx, idx + n_coefficients] += -step_size_diff[idx]
        delta[idx] = (
            coefficients_all[idx, idx] - coefficients_all[idx, idx + n_coefficients]
        )

    ui_replacement_rate_vector_all = np.full(
        (n_types, n_periods_working, n_runs), np.nan
    )
    for run_idx in range(n_runs):
        ui_replacement_rate_vector_tmp = np.full((n_types, n_periods_working), np.nan)
        for type_idx in range(n_types):
            idx_start = ui_replacement_rate_grid_reduced_size * type_idx
            idx_end = ui_replacement_rate_grid_reduced_size * (type_idx + 1)

            ui_replacement_rate_vector_tmp[type_idx, :] = interpolate.PchipInterpolator(
                ui_replacement_rate_grid_reduced,
                coefficients_all[idx_start:idx_end, run_idx],
            )(ui_replacement_rate_grid)
        ui_replacement_rate_vector_tmp = np.maximum(
            ui_replacement_rate_vector_tmp, ui_replacement_rate_min
        )
        ui_replacement_rate_vector_all[:, :, run_idx] = ui_replacement_rate_vector_tmp

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": ui_replacement_rate_vector_all[
                        :, :, run_idx
                    ]
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_workers) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Jacobian",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for coefficient_idx in range(n_coefficients):
        for shock_idx in range(2):
            fx[coefficient_idx, shock_idx] = np.array(
                out[coefficient_idx + n_coefficients * shock_idx][
                    "average_pv_utility_computed_corrected"
                ]
            )

    # compute jacobian
    jacobian = np.full(n_coefficients, np.nan)
    for idx in range(n_coefficients):
        jacobian[idx] = (fx[idx, 0] - fx[idx, 1]) / delta[idx]

    return jacobian


def qnewton(func, jac, x_ini, controls, *args):
    """
    Solve unconstrained maximization problem using quasi-Newton methods.

    :parameter:
    func : functional
        Objective function to maximize.
    jac : functional
        Function that returns function value and Jacobian of
        objective function.
    x_ini : array
        Initial guess for coefficients of local maximum.
    controls : dict
        Dictionary of function controls (details see below)
    *args : tuple
        Additional arguments for objective function.

    :returns:
    x : float
        Coefficients of local maximum of objective function.
    fx : float
        Value of objective function at x
    g : array [len(x) x 1]
        Gradient of objective function at x
    hessian : array [len(x) x len(x)]
        Approximation of the inverse Hessian of the objective function at x.

    :raises:
    ValueError : NaNs or INFs in coefficients.

    The user defined functions FUNC and JAC must have the following syntax
        fx, instr_eq = f(x, controls, *args)
        g = jac(x, controls, *args)
    where, in either case, the additional variables are the ones passed to QNEWTON

    :controls:
        interpolation_method : string,
            Interpolation method for 1D interpolation ("linear" or "cubic")
        n_iterations_jacobian_max : int,
            Maximum number of model_analysis solution iterations for computation of
            jacobian
        n_iterations_opt_max : int,
            Maximum number of iterations of the optimization algorithm
        n_iterations_solve_max : int,
            Maximum number of model_analysis solution iterations for computation of
            value of objective function
        n_iterations_step_max : int,
            Maximum number of iterations of the step search algorithm
        n_simulations : int,
            Number of simulations for model_analysis simulation
        n_workers : int,
            Number of cores used for parallel processing
        run_simulation : bool,
            Flag to activate / deactivate model_analysis simulation
        show_progress : bool,
            Flag to activate / deactivate output of progress bar for gradient
            computation
        show_progress_solve : bool,
            Flag to activate / deactivate output of status updates for model_analysis
            solution
        show_summary : bool,
            Flag to activate / deactivate output of summary statistics for model_analysis
            solution iterations
        step_method : string,
            Step search method ("bt" or "gold")  # todo: adjust after implementation of bhhh
        step_size_jacobian : float,
            Size of disturbance for finite difference calculation in gradient
            computation
        tolerance_solve : float,
            Tolerance for government budget balance in model_analysis solution algorithm
        eps0 : float
            zero factor (used in convergence criteria) (default = 1)
        n_iterations_opt_max : int
            Maximum major iterations (default = 250)
        n_iterations_step_max : int
            Maximum step search iterations (default = 50)
        step_method : str
            Method to calculate optimal step length. Available options
            - "full" : step length is set to 1
            - "bhhh" : BHHH STEP (currently not implemented)
                    # todo: adjust after implementation
            - "bt" : BT STEP (default)
            - "gold" : GOLD STEP (called others fail)
        tol : float
            convergence tolerance (default = sqrt(eps))

    Modified from the corresponding file by Paul L. Fackler & Mario J.Miranda
    paul_fackler@ncsu.edu, miranda.4@osu.edu

    """

    # load controls
    n_iterations_opt_max = controls["n_iterations_opt_max"]
    interpolation_method = controls["interpolation_method"]
    tolerance_bfgs_update = controls["tolerance_bfgs_update"]
    tolerance_convergence_gradient = controls["tolerance_convergence_gradient"]
    tolerance_convergence_marquardt = controls["tolerance_convergence_marquardt"]
    tolerance_slope_min = controls["tolerance_slope_min"]
    zero_factor_convergence_marquardt = controls["zero_factor_convergence_marquardt"]

    # load calibration
    instrument = calibration["instrument"]
    bounds_lower = calibration["bounds_lower"]
    bounds_upper = calibration["bounds_upper"]

    ####################
    # initiate algorithm
    iteration_opt = 0
    k = x_ini.shape[0]
    reset = True
    print(
        "\n###############################################"
        "###############################################\n"
        "QNEWTON: start \n"
        "################################################"
        "##############################################\n"
    )
    print("compute initial function value")
    fx0, instr_eq = func(x_ini, controls, *args)

    # update equilibrium instrument rate
    if instrument == "consumption_tax":
        calibration["consumption_tax_rate_init"][interpolation_method] = instr_eq[
            "tax_consumption"
        ]
        calibration["transfers_pensions_init"] = instr_eq["transfers_pensions"]
        calibration["transfers_lumpsum_init"] = instr_eq["transfers_lumpsum"]
    elif instrument == "income_tax_rate":
        calibration["income_tax_rate_init"][interpolation_method] = instr_eq["tax_ui"]
        calibration["transfers_pensions_init"] = instr_eq["transfers_pensions"]
        calibration["transfers_lumpsum_init"] = instr_eq["transfers_lumpsum"]

    print("compute initial Jacobian")
    g0 = jac(x_ini, controls, *args)
    print(
        "\n###############################################"
        "###############################################\n"
        "QNEWTON: initialization \n"
        "    iteration"
        + " " * (81 - len(f"{iteration_opt:4d}"))
        + f"{iteration_opt:4d}\n"
        "    starting coefficient vector"
        + " " * (63 - len("[" + ", ".join(f"{i:1.5f}" for i in x_ini) + "]"))
        + "["
        + ", ".join(f"{i:1.5f}" for i in x_ini)
        + "]\n"
        "    starting value of objective function"
        + " " * (54 - len(f"{fx0:1.5f}"))
        + f"{fx0:1.5f}\n"
        "    starting gradient norm"
        + " " * (68 - len(f"{np.linalg.norm(g0):9.4f}"))
        + f"{np.linalg.norm(g0):9.4f}\n"
        "################################################"
        "##############################################\n"
    )

    # get approximate hessian
    hessian = -np.identity(k) / max(abs(fx0), 1)

    if np.all(abs(g0) < tolerance_convergence_gradient):
        print("Gradient tolerance reached at starting value")
        return x_ini, fx0, g0, hessian, instr_eq

    ####################
    # start iteration
    x = x_ini
    fx = fx0
    g = g0
    d = 0

    while iteration_opt <= n_iterations_opt_max:

        iteration_opt += 1

        d = -np.dot(hessian, g0)  # search direction

        # if increase in objective in the direction of search is too low,
        # revert to steepest ascent (B = I)
        if np.dot(d, g0) / np.dot(d, d) < tolerance_slope_min:
            hessian = -np.identity(k) / max(abs(fx0), 1)
            d = g0 / max(abs(fx0), 1)
            reset = 1

        print("compute optimal step length")
        s, fx, instr_eq, iterations, err = get_step_size(
            func, x, fx0, g0, d, controls, *args
        )

        # check for step search failure
        if fx <= fx0:
            if reset:  # if already using steepest ascent, break
                warnings.warn("Iterations stuck in qnewton")
                return x, fx0, g0, hessian, instr_eq
            else:  # else, try again with steepest ascent
                hessian = -np.identity(k) / max(abs(fx0), 1)
                d = g0 / max(abs(fx0), 1)
                s, fx, instr_eq, iterations, err = get_step_size(
                    func, x, fx0, g0, d, controls, *args
                )
                if err:
                    warnings.warn("Cannot find suitable step in qnewton")
                    return x, fx0, g0, hessian, instr_eq

        # run some checks, then update step and current coefficient vector
        if np.logical_or(np.any(np.isnan(x + (s * d))), np.any(np.isinf(x + (s * d)))):
            raise ValueError("NaNs or INFs in coefficients.")
        elif np.logical_or(
            np.any(x + (s * d) < bounds_lower), np.any(x + (s * d) > bounds_upper)
        ):
            warnings.warn("Coefficient values out of bounds")
            break
        else:
            d = s * d
            x = x + d

        # update equilibrium instrument rate
        if instrument == "consumption_tax":
            calibration["consumption_tax_rate_init"][interpolation_method] = instr_eq
            calibration["transfers_pensions_init"] = instr_eq["transfers_pensions"]
            calibration["transfers_lumpsum_init"] = instr_eq["transfers_lumpsum"]
        elif instrument == "income_tax_rate":
            calibration["income_tax_rate_init"][interpolation_method] = instr_eq
            calibration["transfers_pensions_init"] = instr_eq["transfers_pensions"]
            calibration["transfers_lumpsum_init"] = instr_eq["transfers_lumpsum"]

        # compute Jacobian
        print("compute jacobian after step")
        g = jac(x, controls, *args)

        print(
            "\n###############################################"
            "###############################################\n"
            "QNEWTON: optimization \n"
            "    iteration"
            + " " * (81 - len(f"{iteration_opt:4d}"))
            + f"{iteration_opt:4d}\n"
            "    current coefficient vector"
            + " " * (64 - len("[" + ", ".join(f"{i:1.5f}" for i in x) + "]"))
            + "["
            + ", ".join(f"{i:1.5f}" for i in x)
            + "]\n"
            "    current value of objective function"
            + " " * (55 - len(f"{fx:1.5f}"))
            + f"{fx:1.5f}\n"
            "    current step norm"
            + " " * (73 - len(f"{np.linalg.norm(d):9.4f}"))
            + f"{np.linalg.norm(d):9.4f}\n"
            "    current gradient norm"
            + " " * (69 - len(f"{np.linalg.norm(g):9.4f}"))
            + f"{np.linalg.norm(g):9.4f}\n"
            "################################################"
            "##############################################\n"
        )

        # test convergence using Marquardt's criterion and gradient test
        if np.logical_or(
            np.logical_and(
                (fx - fx0) / (abs(fx) + zero_factor_convergence_marquardt)
                < tolerance_convergence_marquardt,
                np.all(
                    abs(d) / (abs(x) + zero_factor_convergence_marquardt)
                    < tolerance_convergence_marquardt
                ),
            ),
            np.all(abs(g) < tolerance_convergence_gradient),
        ):
            print("converged")
            break

        # update inverse Hessian approximation
        u = g - g0
        ud = np.dot(u, d)

        # if update could be numerically inaccurate, revert to steepest ascent,
        # otherwise use BFGS update
        if (abs(ud) / (np.linalg.norm(d) * np.linalg.norm(u))) < tolerance_bfgs_update:
            hessian = -np.identity(k) / max(abs(fx), 1)
            reset = True
        else:
            w = d - np.dot(hessian, u)
            wd = np.outer(w, d)
            hessian = (
                hessian + ((wd + wd.T) - (np.dot(u, w) * np.outer(d, d)) / ud) / ud
            )
            reset = False

        # update objects for iteration
        fx0 = fx
        g0 = g

    ####################
    # iteration complete
    if iteration_opt == n_iterations_opt_max:
        warnings.warn("Maximum iterations exceeded in qnewton")

    print(
        "\n###############################################"
        "###############################################\n"
        "QNEWTON: complete \n"
        "    iteration"
        + " " * (81 - len(f"{iteration_opt:4d}"))
        + f"{iteration_opt:4d}\n"
        "    final coefficient vector"
        + " " * (66 - len("[" + ", ".join(f"{i:1.5f}" for i in x) + "]"))
        + "["
        + ", ".join(f"{i:1.5f}" for i in x)
        + "]\n"
        "    final value of objective function"
        + " " * (57 - len(f"{fx:1.5f}"))
        + f"{fx:1.5f}\n"
        "    final step norm"
        + " " * (75 - len(f"{np.linalg.norm(d):9.4f}"))
        + f"{np.linalg.norm(d):9.4f}\n"
        "    final gradient norm"
        + " " * (71 - len(f"{np.linalg.norm(g):9.4f}"))
        + f"{np.linalg.norm(g):9.4f}\n"
        "################################################"
        "##############################################\n"
    )

    return x, fx, g, hessian, instr_eq


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    try:
        setup_name = sys.argv[1]
        method = sys.argv[2]
    except IndexError:
        setup_name = "opt_rate_both_fixed_budget"
        method = "linear"

    # load calibration and set variables
    calibration = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_jacobian_max": 10,
        "n_iterations_opt_max": 50,
        "n_iterations_solve_max": 20,
        "n_iterations_step_max": 20,
        "n_simulations": int(1e4),
        "n_workers": 15,
        "run_simulation": False,
        "seed_simulation": 3405,
        "show_progress": True,
        "show_progress_solve": False,
        "show_summary": False,
        "step_method": "bt",
        "step_size_jacobian": 0.015,
        "tolerance_bfgs_update": 1e-9,
        "tolerance_convergence_gradient": 1e-6,
        "tolerance_convergence_marquardt": 1e-4,
        "tolerance_solve": 1e-3,
        "tolerance_slope_min": 1e-6,
        "zero_factor_convergence_marquardt": 1,
    }

    # load some variables from calibration
    n_periods_working = calibration["n_periods_working"]
    n_types = calibration["n_types"]

    # get starting value
    try:
        x_ini = np.array(calibration["ui_params_opt_init"])
    except KeyError:
        if "rate_only" in setup_name:
            x_ini = np.array([0.5])
        elif "rate_age" in setup_name:
            x_ini = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        elif "rate_type" in setup_name:
            x_ini = np.array([0.5, 0.5, 0.5])
        elif "rate_both" in setup_name:
            x_ini = np.array(
                [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ],
            )
        elif "rate_floor_cap" in setup_name:
            x_ini = np.array([0.5, 0.2, 0.8])
        else:
            raise ValueError(
                "Scope of optimization unclear; "
                "please select setup name containing one of "
                "['rate_only', 'rate_age', 'rate_type', 'rate_both', 'rate_floor_cap']"
            )

    # update calibration
    if "rate_only" in setup_name:
        calibration["ui_replacement_rate_vector"] = np.full(
            (n_types, n_periods_working), x_ini[0]
        ).tolist()
        calibration["bounds_lower"] = np.array([0.0])
        calibration["bounds_upper"] = np.array([1.0])
    elif "rate_age" in setup_name:
        ui_grid = np.array(calibration["ui_grid"])
        ui_replacement_rate_min = calibration["ui_replacement_rate_min"]
        ui_vector = interpolate.PchipInterpolator(ui_grid, x_ini)(
            np.linspace(0, n_periods_working - 1, n_periods_working)
        )
        ui_vector = np.maximum(ui_vector, ui_replacement_rate_min)
        ui_vector = np.tile(ui_vector, n_types).reshape((n_types, n_periods_working))
        calibration["ui_replacement_rate_vector"] = ui_vector.tolist()
        calibration["bounds_lower"] = np.full(len(x_ini), -0.2)
        calibration["bounds_upper"] = np.full(len(x_ini), 1.0)
    elif "rate_type" in setup_name:
        calibration["ui_replacement_rate_vector"] = np.repeat(
            x_ini, n_periods_working
        ).reshape((n_types, n_periods_working))
        calibration["bounds_lower"] = np.array([0.0, 0.0, 0.0])
        calibration["bounds_upper"] = np.array([1.0, 1.0, 1.0])
    elif "rate_both" in setup_name:
        ui_grid = np.array(calibration["ui_grid"])
        ui_replacement_rate_min = calibration["ui_replacement_rate_min"]
        ui_vector = np.full((n_types, n_periods_working), np.nan)
        for type_idx in range(n_types):
            ui_vector[type_idx, :] = interpolate.PchipInterpolator(
                ui_grid,
                x_ini[len(ui_grid) * type_idx : len(ui_grid) * (type_idx + 1)],
            )(np.linspace(0, n_periods_working - 1, n_periods_working))
        ui_vector = np.maximum(ui_vector, ui_replacement_rate_min)
        calibration["ui_replacement_rate_vector"] = ui_vector.tolist()
        calibration["bounds_lower"] = np.full((n_types, len(x_ini)), -0.25)
        calibration["bounds_upper"] = np.full((n_types, len(x_ini)), 1.2)
    elif "rate_floor_cap" in setup_name:
        calibration["ui_replacement_rate_vector"] = np.full(
            (n_types, n_periods_working), x_ini[0]
        ).tolist()
        calibration["ui_floor"] = x_ini[1]
        calibration["ui_cap"] = x_ini[2]
        calibration["bounds_lower"] = np.array([0.0, 0.0, 0.0])
        calibration["bounds_upper"] = np.array([1.0, 2.0, 2.0])
    else:
        raise ValueError(
            "Scope of optimization unclear; "
            "please select setup name containing one of "
            "['rate_only', 'rate_age', 'rate_type', 'rate_both', 'rate_floor_cap']"
        )

    # optimize
    if "rate_only" in setup_name:
        func = _eval_rate
        jac = _jacobian_rate
    elif "rate_age" in setup_name:
        func = _eval_rate_vector
        jac = _jacobian_rate_vector
    elif "rate_type" in setup_name:
        func = _eval_rate_type
        jac = _jacobian_rate_type
    elif "rate_both" in setup_name:
        func = _eval_rate_age_type
        jac = _jacobian_rate_age_type
    elif "rate_floor_cap" in setup_name:
        func = _eval_rate_floor_cap
        jac = _jacobian_rate_floor_cap
    else:
        raise ValueError(
            "Scope of optimization unclear; "
            "please select setup name containing one of "
            "['rate_only', 'rate_age', 'rate_type', 'rate_both', 'rate_floor_cap']"
        )

    # run optimization
    x_opt, fx_opt, g_opt, hessian, instr_eq = qnewton(
        func, jac, x_ini, controls, calibration
    )

    # compile & store results
    results = {
        "optimization": method,
        "ui_coefficients_opt": x_opt,
        "welfare_opt": fx_opt,
        "equilibrium_instrument_rate": instr_eq,
    }

    for item in results:
        try:
            results[item] = results[item].tolist()
        except AttributeError:
            pass

    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_" + setup_name + "_optimization_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)
