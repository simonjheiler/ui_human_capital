import copy
import json  # noqa:F401
import src.utilities.istarmap_3_8  # noqa, noreorder
import multiprocessing
import sys  # noqa:F401
import tqdm  # noqa:F401
import warnings  # noqa:F401

import numba as nb  # noqa:F401
import numpy as np  # noqa:F401
import pandas as pd  # noqa:F401
from scipy import interpolate  # noqa:F401

from bld.project_paths import project_paths_join as ppj  # noqa:F401
from src.model_analysis.run_utils import _solve_run
from src.utilities.optimization_utils import get_step_size

#####################################################
# PARAMETERS
#####################################################

age_thresholds_urate = [-np.inf, 21, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
age_groups_urate = [
    "20",
    "21 to 24",
    "25 to 29",
    "30 to 34",
    "35 to 39",
    "40 to 44",
    "45 to 49",
    "50 to 54",
    "55 to 59",
    "60 and older",
]


#####################################################
# FUNCTIONS
#####################################################


def _average_by_age_group(array_in, age_min, thresholds, labels):

    array_in.loc[:, "age"] = array_in.index // 4 + age_min
    array_in.loc[:, "age_group"] = pd.cut(
        array_in.age, thresholds, right=False, labels=labels
    )
    array_out = array_in.groupby("age_group").mean()
    array_out = array_out.drop(columns="age")

    return array_out


def _eval_fit(coefficients, controls, calibration):
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
    age_min = calibration["age_min"]
    leisure_base = np.array(calibration["leisure_base"])
    leisure_grid = np.array(calibration["leisure_grid"])
    type_weights = np.array(calibration["type_weights"])

    # get derivative of leisure utility from coefficients
    leisure_utility_new = interpolate.PchipInterpolator(
        leisure_base, coefficients
    ).antiderivative()(leisure_grid)
    leisure_utility_dx_new = interpolate.PchipInterpolator(leisure_base, coefficients)(
        leisure_grid
    )
    leisure_utility_dxdx_new = interpolate.PchipInterpolator(
        leisure_base, coefficients
    ).derivative()(leisure_grid)

    # update calibration
    calibration["leisure_utility"] = leisure_utility_new.tolist()
    calibration["leisure_utility_dx"] = leisure_utility_dx_new.tolist()
    calibration["leisure_utility_dxdx"] = leisure_utility_dxdx_new.tolist()

    # solve model_analysis
    results = _solve_run({}, controls, calibration)

    # extract outputs
    share_nonemployed_mean = pd.DataFrame(
        np.array(results["share_nonemployed"]).T, columns=["high", "medium", "low"]
    )
    share_searching_mean = pd.DataFrame(
        np.array(results["share_searching"]).T, columns=["high", "medium", "low"]
    )
    unemployment_rate = (share_searching_mean + share_nonemployed_mean) / 2

    job_finding_probability_mean = pd.DataFrame(
        np.array(results["job_finding_probability_searching_all_mean"]).T,
        columns=["high", "medium", "low"],
    )
    job_finding_rate_mean = pd.DataFrame(
        np.array(results["job_finding_rate_searching_all_mean"]).T,
        columns=["high", "medium", "low"],
    )
    equilibrium_instrument_rate = results["equilibrium_instrument_rate"]

    unemployment_rate_by_age_group = _average_by_age_group(
        unemployment_rate, age_min, age_thresholds_urate, age_groups_urate
    )
    job_finding_probability_by_age_group = _average_by_age_group(
        job_finding_probability_mean, age_min, age_thresholds_urate, age_groups_urate
    )
    job_finding_rate_by_age_group = _average_by_age_group(
        job_finding_rate_mean, age_min, age_thresholds_urate, age_groups_urate
    )

    unemployment_rate_by_age_group = unemployment_rate_by_age_group.drop(
        ["20", "60 and older"]
    )
    job_finding_probability_by_age_group = job_finding_probability_by_age_group.drop(
        ["20", "60 and older"]
    )
    job_finding_rate_by_age_group = job_finding_rate_by_age_group.drop(
        ["20", "60 and older"]
    )

    # compute objective for MAXIMIZATION
    fit = np.average(
        np.sqrt(
            np.sum(np.square(unemployment_rate_by_age_group - target_unemployment))
        ),
        weights=type_weights,
    )
    objective = -fit

    return objective, equilibrium_instrument_rate


def _jacobian_fit(coefficients, controls, calibration):
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
    age_min = calibration["age_min"]
    leisure_base = np.array(calibration["leisure_base"])
    leisure_grid = np.array(calibration["leisure_grid"])
    type_weights = np.array(calibration["type_weights"])

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

    leisure_utility_dx_all = np.full((len(leisure_grid), n_runs), np.nan)
    for run_idx in range(n_runs):
        leisure_utility_dx_tmp = interpolate.PchipInterpolator(
            leisure_base, coefficients_all[:, run_idx]
        )(leisure_grid)
        leisure_utility_dx_tmp = np.minimum(leisure_utility_dx_tmp, 0.0)
        leisure_utility_dx_all[:, run_idx] = leisure_utility_dx_tmp

    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {"leisure_utility_dx": leisure_utility_dx_all[:, run_idx]},
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
        share_nonemployed_mean = pd.DataFrame(
            np.array(out[run_idx]["share_nonemployed"]).T,
            columns=["high", "medium", "low"],
        )
        share_searching_mean = pd.DataFrame(
            np.array(out[run_idx]["share_searching"]).T,
            columns=["high", "medium", "low"],
        )
        unemployment_rate = (share_searching_mean + share_nonemployed_mean) / 2
        unemployment_rate_by_age_group = _average_by_age_group(
            unemployment_rate, age_min, age_thresholds_urate, age_groups_urate
        )
        unemployment_rate_by_age_group = unemployment_rate_by_age_group.drop(
            ["20", "60 and older"]
        )
        fit = np.average(
            np.sqrt(
                np.sum(np.square(unemployment_rate_by_age_group - target_unemployment))
            ),
            weights=type_weights,
        )

        fx[run_idx] = -fit

    # reshape
    fx = np.moveaxis(np.stack((fx[:n_coefficients], fx[n_coefficients:])), 0, -1)

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
        calibration["consumption_tax_rate_init"][interpolation_method] = instr_eq
    elif instrument == "income_tax_rate":
        calibration["income_tax_rate_init"][interpolation_method] = instr_eq

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
        elif instrument == "income_tax_rate":
            calibration["income_tax_rate_init"][interpolation_method] = instr_eq

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

    # load calibration and set variables
    calibration = json.load(
        open(
            ppj(
                "IN_MODEL_SPECS",
                "analytics_calibration_base_combined_inctax.json",
            )
        )
    )

    # set controls
    controls = {
        "interpolation_method": "linear",
        "n_iterations_jacobian_max": 10,
        "n_iterations_opt_max": 50,
        "n_iterations_solve_max": 20,
        "n_iterations_step_max": 20,
        "n_simulations": int(1e5),
        "n_workers": 15,
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_progress": True,
        "show_progress_solve": False,
        "show_summary": False,
        "step_method": "bt",
        "step_size_jacobian": 0.025,
        "tolerance_bfgs_update": 1e-9,
        "tolerance_convergence_gradient": 1e-6,
        "tolerance_convergence_marquardt": 1e-4,
        "tolerance_solve": 1e-5,
        "tolerance_slope_min": 1e-6,
        "zero_factor_convergence_marquardt": 1,
    }

    # load calibration target
    targets_transitions = pd.read_csv(
        ppj("OUT_RESULTS", "empirics", "cps_transition_probabilities.csv"),
        index_col=["age_group", "education_reduced"],
    )
    targets_unemployment = pd.read_csv(
        ppj("OUT_RESULTS", "empirics", "cps_unemployment_probabilities.csv"),
        index_col=["age_group", "education_reduced"],
    )
    target_unemployment = targets_unemployment.loc[:, "estimate"].unstack(level=1)
    target_finding = targets_transitions.loc[:, "p_ue_3m_computed"].unstack(level=1)
    target_unemployment = target_unemployment.drop("60 to 64")
    target_finding = target_finding.drop("60 to 64")

    # set starting point for optimization
    x_ini = np.array(
        [
            -0.38058201782797624,
            -0.4363001134658238,
            -2.779983793878501,
            -11.064102042328775,
            -349.55882,
        ]
        # [
        #     -0.44035,
        #     -0.58798,
        #     -2.91690,
        #     -11.06073,
        #     -349.55882
        # ]
        # [
        #     -0.43814176351636497,
        #     -0.5988137997537946,
        #     -2.9211951315288576,
        #     -11.060335732038416,
        #     -349.55881966924466,
        # ]
        # [
        #     -0.41704,
        #     -0.70502,
        #     -3.02191,
        #     -11.05694,
        #     -349.55882
        # ]
        # [
        #     -0.44405,
        #     -0.47981,
        #     -3.09841,
        #     -11.06310,
        #     -349.55882
        # ]
        # [
        #     -0.3835288573545746,
        #     -1.2516126353720638,
        #     -2.2047056582301963,
        #     -11.035550211288463,
        #     -349.5588394449981
        # ]
    )

    # adjust calibration
    leisure_grid = np.linspace(0, 1, 1001)
    leisure_base = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    leisure_utility_dx_new = interpolate.PchipInterpolator(
        leisure_base, x_ini, extrapolate=True
    )(leisure_grid)
    calibration["leisure_utility_dx"] = leisure_utility_dx_new.tolist()
    calibration["leisure_base"] = leisure_base.tolist()
    calibration["leisure_grid"] = leisure_grid.tolist()
    calibration["bounds_lower"] = [-np.inf] * len(leisure_base)
    calibration["bounds_upper"] = [0.0] * len(leisure_base)

    # run optimization
    x_opt, fx_opt, g_opt, hessian, instr_eq = qnewton(
        _eval_fit, _jacobian_fit, x_ini, controls, calibration
    )

    leisure_utility_dx_interpolator = interpolate.PchipInterpolator(
        leisure_base, x_opt, extrapolate=True
    )

    leisure_utility_new = leisure_utility_dx_interpolator.antiderivative()(leisure_grid)
    leisure_utility_dx_new = leisure_utility_dx_interpolator(leisure_grid)
    leisure_utility_dxdx_new = leisure_utility_dx_interpolator.derivative()(
        leisure_grid
    )

    print("pause")
