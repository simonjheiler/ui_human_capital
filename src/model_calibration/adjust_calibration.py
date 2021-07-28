import copy
import json
import src.utilities.istarmap_3_8  # noqa, noreorder
import multiprocessing
import sys  # noqa:F401
import tqdm  # noqa:F401
import warnings  # noqa:F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import optimize

from bld.project_paths import project_paths_join as ppj
from src.model_analysis.run_utils import _solve_run
from src.utilities.optimization_utils import get_step_size
from src.utilities.plot_utils import _average_by_age_group


#####################################################
# PARAMETERS
#####################################################

age_thresholds = [
    -np.inf,
    24,
    29,
    34,
    39,
    44,
    49,
    54,
    59,
    np.inf,
]
age_groups = [
    "20 to 24",
    "25 to 29",
    "30 to 34",
    "35 to 39",
    "40 to 44",
    "45 to 49",
    "50 to 54",
    "55 to 59",
    "60 to 64",
]

age_thresholds_full = [-np.inf, 21, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
age_groups_full = [
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
age_thresholds_edu = [-np.inf, 30, 35, 40, 45, 50, 55, 60, np.inf]
age_groups_edu = [
    "25 to 29",
    "30 to 34",
    "35 to 39",
    "40 to 44",
    "45 to 49",
    "50 to 54",
    "55 to 59",
    "60 and older",
]

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

df_age_coefficients_full = pd.read_csv(
    ppj("OUT_RESULTS", "empirics", "cps_age_coefficients_full.csv"), index_col=0
)
df_age_coefficients_edu_reduced = pd.read_csv(
    ppj("OUT_RESULTS", "empirics", "cps_age_coefficients_edu_reduced.csv"),
    index_col=0,
)

transition_statistics_mean = pd.read_csv(
    ppj("OUT_RESULTS", "empirics", "cps_transition_statistics.csv"),
    index_col=["age_group", "education_group"],
)

transition_statistics_estimated = {}
for statistic in [
    "1m_eu_effects",
    "1m_eu_emmeans",
    "3m_eu_effects",
    "3m_eu_emmeans",
    "1m_ue_effects",
    "1m_ue_emmeans",
    "3m_ue_effects",
    "3m_ue_emmeans",
]:
    tmp = pd.read_csv(
        ppj(
            "OUT_RESULTS",
            "empirics",
            "cps_transition_probability_" + statistic + ".csv",
        ),
        index_col=["x", "group"],
    )
    tmp = tmp.rename(
        columns={
            "predicted": "prediction",
            "std.error": "std",
            "conf.low": "ci_lower",
            "conf.high": "ci_upper",
        },
    )
    tmp.index = tmp.index.rename(["age_group", "education_group"])
    tmp.index = pd.MultiIndex.from_frame(tmp.index.to_frame().replace("1", "overall"))
    transition_statistics_estimated[statistic] = tmp

transition_statistics_estimated_mean = pd.concat(
    [
        transition_statistics_estimated[i].prediction.rename(i)
        for i in transition_statistics_estimated.keys()
    ],
    axis=1,
)

calibration_old = json.load(
    open(ppj("IN_MODEL_SPECS", "analytics_calibration_base_flat_no_caps.json"))
)

targets_transitions = pd.read_csv(
    ppj("OUT_RESULTS", "empirics", "cps_transition_probabilities.csv"),
    index_col=["age_group", "education_reduced"],
)
targets_unemployment = pd.read_csv(
    ppj("OUT_RESULTS", "empirics", "cps_unemployment_probabilities.csv"),
    index_col=["age_group", "education_reduced"],
)

pd.options.display.max_colwidth = 100

global tck


#####################################################
# FUNCTIONS
#####################################################


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
    leisure_utility_dx_new = interpolate.PchipInterpolator(leisure_base, coefficients)(
        leisure_grid
    )

    # update calibration
    calibration["leisure_utility_dx"] = leisure_utility_dx_new.tolist()

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


def _sse(coefficients, base_values, target):
    """
    Compute sum of squared differences between cubic spline through *coefficients* and *target*.
    """
    # compute spline
    x = np.array(range(0, 181, 20))
    coefficients = np.pad(
        coefficients, (0, 2), "constant", constant_values=(coefficients[-1])
    )

    delta_fit = interpolate.PchipInterpolator(x, coefficients)

    # compute sum of squared differences
    sse = np.sum(np.square(delta_fit(base_values) - target))

    return sse


def _calibrate_separations(statistic):

    # define interpolation base
    x_range = np.arange(0, 180)
    x_val = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # load calibration targets
    target = targets_transitions.loc[:, "p_eu_3m_computed"].unstack(level=1)

    # interpolate separation rates
    separation_rates = pd.DataFrame(index=x_range)
    for var in ["low", "medium", "high"]:
        separation_rates.loc[:, var] = interpolate.PchipInterpolator(
            x_val, target.loc[:, var]
        )(x_range)

    # calculate averages by age group
    separation_rates["age"] = separation_rates.index // 4 + 20
    separation_rates["age_group"] = pd.cut(
        separation_rates.loc[:, "age"],
        age_thresholds,
        right=True,
        labels=age_groups,
    )
    separation_rates = separation_rates.drop("age", axis=1)
    separation_rates_mean = separation_rates.groupby("age_group").mean()

    # plot calibrated values vs. targets
    for split in ["low", "medium", "high"]:

        # compare vector with target
        fig, ax = plt.subplots()

        ax.plot(x_val, separation_rates_mean.loc[:, split])
        ax.plot(x_val, target.loc[:, split])

        fig.savefig(
            ppj(
                "OUT_FIGURES",
                "sandbox",
                "calibration_separation_rates_fit_mean_" + split + ".pdf",
            )
        )
        plt.close()

        # compare means with targets
        fig, ax = plt.subplots()

        ax.plot(x_range, separation_rates.loc[:, split])
        ax.plot(x_val, target.loc[:, split])

        fig.savefig(
            ppj(
                "OUT_FIGURES",
                "sandbox",
                "calibration_separation_rates_fit_vector_" + split + ".pdf",
            )
        )
        plt.close()

    # collect output
    separations_out = {
        "low": separation_rates.loc[:, "low"].tolist(),
        "medium": separation_rates.loc[:, "medium"].tolist(),
        "high": separation_rates.loc[:, "high"].tolist(),
    }

    return separations_out, target


def _calibrate_wages():

    type_weights = np.array([0.3118648, 0.5777581, 0.1103771])

    # CONSTRUCT TARGETS

    # no split by education
    targets_full = df_age_coefficients_full
    targets_full.loc[:, "age_group"] = pd.cut(
        targets_full.index, age_thresholds_full, right=False, labels=age_groups_full
    )
    targets_full.coefficient = np.exp(targets_full.coefficient)
    targets_full["weighted_coefficient"] = (
        targets_full.coefficient * targets_full.weight
    )
    targets_full = targets_full.groupby(["age_group"])[
        ["weighted_coefficient", "weight"]
    ].sum()
    targets_full["coefficient"] = (
        targets_full.weighted_coefficient / targets_full.weight
    )
    targets_full.loc["20", "coefficient"] = 1.0
    targets_full = targets_full.rename(columns={"coefficient": "overall"})

    # split by education
    targets_edu = df_age_coefficients_edu_reduced
    targets_edu.loc[:, "age_group"] = pd.cut(
        targets_edu.index, age_thresholds_edu, right=False, labels=age_groups_edu
    )
    targets_edu.coefficient = np.exp(targets_edu.coefficient)
    targets_edu["weighted_coefficient"] = targets_edu.coefficient * targets_edu.weight
    targets_edu = targets_edu.groupby(["age_group", "type"])[
        ["weighted_coefficient", "weight"]
    ].sum()
    targets_edu["coefficient"] = targets_edu.weighted_coefficient / targets_edu.weight
    targets_edu = targets_edu["coefficient"].unstack()

    # combine data frames
    targets = pd.merge(
        targets_full["overall"],
        targets_edu,
        left_index=True,
        right_index=True,
        how="left",
    )
    targets = targets[["overall", "high", "medium", "low"]]

    # construct wage level by type for ages 20 to 24
    # compute average wage growth
    factors = targets.iloc[:2, 0] / targets.iloc[2, 0]

    targets.iloc[:2, 1:] = (
        np.tile(targets.loc["25 to 29", ["high", "medium", "low"]], 2).reshape((2, 3))
        * np.tile(factors, 3).reshape((3, 2)).T
    )

    # adjust s.t. aggregate wage at 20 = 1.0
    scaling = np.average(
        targets.loc["20", ["high", "medium", "low"]], weights=type_weights
    )
    targets.iloc[:, 1:] = targets.iloc[:, 1:] / scaling

    # CALIBRATION

    # load setup
    setup_name = "base_combined_recalibrated"
    method = "linear"

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_solve_max": 20,
        "n_simulations": int(1e4),
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_progress_solve": True,
        "show_summary": True,
        "tolerance_solve": 1e-4,
    }

    # load calibration and set some variables
    calibration_old = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )
    age_min = calibration_old["age_min"]
    wage_in = pd.DataFrame(
        np.array(calibration_old["wage_hc_factor_vector"]).T,
        columns=["high", "medium", "low"],
    )
    wage_in.loc[:, "overall"] = np.average(
        wage_in.loc[:, ["high", "medium", "low"]], weights=type_weights, axis=1
    )

    # simulate economy
    results = _solve_run({}, controls, calibration_old)

    # extract simulated hc levels and wage levels
    experience_mean = pd.DataFrame(
        np.array(results["hc_employed_mean"]).T,
        columns=["high", "medium", "low"],
    )
    wage_mean = pd.DataFrame(
        np.array(results["wage_hc_factor_employed_mean"]).T,
        columns=["high", "medium", "low"],
    )

    # construct statistics
    experience_mean.loc[:, "overall"] = np.average(
        experience_mean.loc[:, ["high", "medium", "low"]], weights=type_weights, axis=1
    )
    experience_by_age_group = _average_by_age_group(
        experience_mean, age_min, age_thresholds_full, age_groups_full
    )
    experience_by_age_group.iloc[0, :] = 0.0

    wage_mean.loc[:, "overall"] = np.average(
        wage_mean.loc[:, ["high", "medium", "low"]], weights=type_weights, axis=1
    )
    wage_by_age_group = _average_by_age_group(
        wage_mean, age_min, age_thresholds_full, age_groups_full
    )

    # create some plots
    x_range = np.arange(0, 181)
    x_val = np.array([20, 22.5, 27, 32, 37, 42, 47, 52, 57, 62])

    for split in ["overall", "high", "medium", "low"]:
        fig, ax = plt.subplots()
        ax.plot(x_val, wage_by_age_group.loc[:, split])
        ax.plot(x_val, targets.loc[:, split])
        fig.savefig(
            ppj("OUT_FIGURES", "sandbox", "calibration_wages_fit_" + split + ".pdf")
        )
        plt.close()

    # fit wage hc vectors
    x_ini = np.arange(1.0, 1.8, 0.1)
    cons = tuple(
        {"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]} for i in range(7)
    )

    wage_hc_factor_new = pd.DataFrame(index=np.arange(0, 181))
    x_opt = pd.DataFrame(index=np.arange(0, 8))

    for split in ["overall", "high", "medium", "low"]:
        # fit wage hc curve to minimize difference at simulated
        # hc levels at targets
        opt = optimize.minimize(
            _sse,
            x_ini,
            args=(
                experience_by_age_group.loc[:, split],
                targets.loc[:, split],
            ),
            constraints=cons,
        )
        x_opt.loc[:, split] = opt.x

        # interpolate complete wage hc vector
        wage_hc_factor_new.loc[:, split] = interpolate.PchipInterpolator(
            np.arange(0, 181, 20),
            np.pad(opt.x, (0, 2), "constant", constant_values=(opt.x[-1])),
        )(x_range)

    # plot for visual comparison
    for split in ["overall", "high", "medium", "low"]:
        # wage in vs. wage out
        fig, ax = plt.subplots()
        ax.plot(x_range, wage_in.loc[:, split])
        ax.plot(x_range, wage_hc_factor_new.loc[:, split])
        fig.savefig(
            ppj(
                "OUT_FIGURES",
                "sandbox",
                "calibration_wages_fit_in_out_" + split + ".pdf",
            )
        )
        plt.close()

        # wage out vs targets
        fig, ax = plt.subplots()
        ax.plot((x_val - 20) * 4, targets.loc[:, split])
        ax.plot(x_range, wage_hc_factor_new.loc[:, split])
        fig.savefig(
            ppj(
                "OUT_FIGURES",
                "sandbox",
                "calibration_wages_fit_out_target_" + split + ".pdf",
            )
        )
        plt.close()

    wage_hc_factor_out = {
        "overall": wage_hc_factor_new.loc[:, "overall"].tolist(),
        "low": wage_hc_factor_new.loc[:, "low"].tolist(),
        "medium": wage_hc_factor_new.loc[:, "medium"].tolist(),
        "high": wage_hc_factor_new.loc[:, "high"].tolist(),
    }

    return wage_hc_factor_out, targets


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    # load calibration and set variables
    calibration = json.load(
        open(
            ppj(
                "IN_MODEL_SPECS",
                "analytics_calibration_base_combined_recalibrated_no_inctax.json",
            )
        )
    )

    # calibrate leisure utility function

    # set some variables
    leisure_grid = np.linspace(0, 1, 1001)
    leisure_base = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # set starting point for optimization
    x_ini = np.array(
        [
            -0.3820180873012038,
            -0.417167939474483,
            -3.096225236446147,
            -11.067836992907717,
            -349.5588188146288,
        ]
    )

    run_optimization = False

    if run_optimization:
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

        # adjust calibration
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
    else:
        x_opt = x_ini

    leisure_utility_dx_interpolator = interpolate.PchipInterpolator(
        leisure_base, x_opt, extrapolate=True
    )

    leisure_utility_new = leisure_utility_dx_interpolator.antiderivative()(leisure_grid)
    leisure_utility_dx_new = leisure_utility_dx_interpolator(leisure_grid)
    leisure_utility_dxdx_new = leisure_utility_dx_interpolator.derivative()(
        leisure_grid
    )

    leisure_utility_functions = {
        "leisure_utility": leisure_utility_new.tolist(),
        "leisure_utility_dx": leisure_utility_dx_new.tolist(),
        "leisure_utility_dxdx": leisure_utility_dxdx_new.tolist(),
    }

    # run calibration
    separations_new, targets_separations = _calibrate_separations(
        "transition_probability_3m_eu"
    )

    wage_hc_factors_new, targets_wages = _calibrate_wages()

    # store results
    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "calibration_separations_new.json",
        ),
        "w",
    ) as outfile:
        json.dump(separations_new, outfile, ensure_ascii=False, indent=2)

    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "calibration_wage_hc_factors_new.json",
        ),
        "w",
    ) as outfile:
        json.dump(wage_hc_factors_new, outfile, ensure_ascii=False, indent=2)

    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "calibration_leisure_utility_functions_no_inctax.json",
        ),
        "w",
    ) as outfile:
        json.dump(leisure_utility_functions, outfile, ensure_ascii=False, indent=2)

    targets_separations.to_csv(
        ppj("OUT_RESULTS", "empirics", "calibration_targets_separations.csv")
    )
    targets_wages.to_csv(
        ppj("OUT_RESULTS", "empirics", "calibration_targets_wages.csv")
    )
