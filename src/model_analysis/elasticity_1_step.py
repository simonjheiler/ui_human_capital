"""
Compute approximate elasticity of unemployment to pension benefits by age.
 by simulation runs the solution routine (see solve_model_old2.py)
 with N=10000; this code can be used in the calibration procedure.
 See elasticity_exact.py for a more precise computation of elasticity by age.
"""
#####################################################
# IMPORTS
#####################################################
import copy
import json
import multiprocessing
import sys

import numpy as np
from bld.project_paths import project_paths_join as ppj

from src.model_analysis.run_utils import _solve_run


#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def elasticity_1_step(controls, calibration):

    # set controls
    n_runs = 3
    n_parallel_jobs = controls["n_parallel_jobs"]
    shock_size = 0.05

    # load variables
    n_periods_working = calibration["n_periods_working"]
    n_types = calibration["n_types"]
    type_weights = np.array(calibration["type_weights"])
    ui_replacement_rate_vector = np.array(calibration["ui_replacement_rate_vector"])

    # calculate derived variables
    n_years_working = int(n_periods_working / 4)

    # initialize objects
    value_at_birth = np.full((n_types, n_runs), np.nan)
    share_nonemployed = np.full((n_types, n_periods_working, n_runs), np.nan)

    # generate shocked input vectors
    shock_direction = np.array([-1, 0, 1])
    shock_timing = np.zeros((n_types, n_periods_working))
    direction = 1
    for year_idx in range(0, n_years_working, 5):
        period_idx_start = int(year_idx * 4)
        period_idx_end = int(min(period_idx_start + 4, n_periods_working))
        shock_timing[:, period_idx_start:period_idx_end] = np.full(4, direction)
        direction = -direction  # invert direction of shock for every bracket

    ui_replacement_rate_vector_all = np.repeat(
        ui_replacement_rate_vector, n_runs
    ).reshape((n_types, n_periods_working, n_runs))
    for run_idx in range(n_runs):
        ui_replacement_rate_vector_all[:, :, run_idx] += (
            shock_timing * 0.5 * shock_size * shock_direction[run_idx]
        )

    # define program for parallel computation
    inputs = []
    for run_idx in range(n_runs):
        inputs += [
            (
                {
                    "ui_replacement_rate_vector": ui_replacement_rate_vector_all[
                        :, :, run_idx
                    ].tolist()
                },
                copy.deepcopy(controls),
                copy.deepcopy(calibration),
            )
        ]

    # solve for all runs of the program (in parallel)
    with multiprocessing.Pool(n_parallel_jobs) as pool:
        out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(n_runs):
        value_at_birth[:, run_idx] = np.array(out[run_idx]["welfare"])
        share_nonemployed[:, :, run_idx] = np.array(out[run_idx]["share_nonemployed"])

    # average over types
    average_value_at_birth = np.average(value_at_birth, weights=type_weights, axis=0)
    average_share_nonemployed = np.average(
        share_nonemployed, weights=type_weights, axis=0
    )
    average_ui_replacement_rate = np.average(
        ui_replacement_rate_vector, weights=type_weights, axis=0
    )

    # calculate elasticities
    age_min = 20
    age_bins = [[20, 30], [35, 45], [50, 60]]
    n_bins = len(age_bins)

    # averaged over types
    average_elasticity_unemployment_up = (
        (average_share_nonemployed[:, 2] - average_share_nonemployed[:, 1])
        * average_ui_replacement_rate
        / (average_share_nonemployed[:, 1] * 0.5 * shock_size * shock_timing[0, :])
    )
    average_elasticity_unemployment_down = (
        (average_share_nonemployed[:, 0] - average_share_nonemployed[:, 1])
        * average_ui_replacement_rate
        / (
            average_share_nonemployed[:, 1]
            * 0.5
            * shock_size
            * shock_timing[0, :]
            * (-1)
        )
    )
    average_elasticity_unemployment = (
        0.5 * average_elasticity_unemployment_down
        + 0.5 * average_elasticity_unemployment_up
    )
    average_elasticity_unemployment = average_elasticity_unemployment * (
        1 - np.isinf(average_elasticity_unemployment)
    )

    average_elasticity_unemployment_mean = np.full(n_bins, np.nan)
    for bin_idx, age_bin in enumerate(age_bins):
        average_elasticity_unemployment_mean[bin_idx] = np.nansum(
            average_elasticity_unemployment[
                (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4
            ]
            * average_share_nonemployed[
                (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4, 2
            ]
        ) / np.sum(
            average_share_nonemployed[
                (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4, 2
            ]
            * (
                1
                - (
                    shock_timing[
                        0, (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4
                    ]
                    == 0
                )
            )
        )

    # by type
    elasticity_unemployment_up = (
        (share_nonemployed[:, :, 2] - share_nonemployed[:, :, 1])
        * ui_replacement_rate_vector
        / (share_nonemployed[:, :, 1] * 0.5 * shock_size * shock_timing)
    )
    elasticity_unemployment_down = (
        (share_nonemployed[:, :, 0] - share_nonemployed[:, :, 1])
        * ui_replacement_rate_vector
        / (share_nonemployed[:, :, 1] * 0.5 * shock_size * shock_timing * (-1))
    )
    elasticity_unemployment = (
        0.5 * elasticity_unemployment_down + 0.5 * elasticity_unemployment_up
    )
    elasticity_unemployment = elasticity_unemployment * (
        1 - np.isinf(elasticity_unemployment)
    )

    elasticity_unemployment_mean = np.full((n_types, n_bins), np.nan)
    for bin_idx, age_bin in enumerate(age_bins):
        elasticity_unemployment_mean[:, bin_idx] = np.nansum(
            elasticity_unemployment[
                :, (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4
            ]
            * share_nonemployed[
                :, (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4, 2
            ],
            axis=1,
        ) / np.sum(
            share_nonemployed[
                :, (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4, 2
            ]
            * (
                1
                - (
                    shock_timing[
                        :, (age_bin[0] - age_min) * 4 : (age_bin[1] - age_min + 1) * 4
                    ]
                    == 0
                )
            ),
            axis=1,
        )

    # print and return results
    print(
        "=====================================\n" " unemployment elasticity to benefits"
    )
    for bin_idx, age_bin in enumerate(age_bins):
        print(
            " {}-{} \t\t\t\t\t   {:9.2f}".format(
                age_bin[0], age_bin[1], average_elasticity_unemployment_mean[bin_idx]
            )
        )
    print("=====================================")

    out = {
        "age_bins": age_bins,
        "average_elasticity_unemployment_mean": average_elasticity_unemployment_mean,
        "average_pv_utility_simulated_corrected": average_value_at_birth,
        "average_share_nonemployed": average_share_nonemployed,
        "elasticity_unemployment_mean": elasticity_unemployment_mean,
        "pv_utility_simulated_corrected": value_at_birth,
        "share_nonemployed": share_nonemployed,
        "ui_replacement_rate_vector_all": ui_replacement_rate_vector_all,
    }

    for item in out:
        try:
            out[item] = out[item].tolist()
        except AttributeError:
            pass

    return out


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    try:
        setup_name = sys.argv[1]
        method = sys.argv[2]
    except IndexError:
        setup_name = "base_combined"
        method = "linear"

    # load calibration and set some variables
    calibration = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_solve_max": 20,
        "n_simulations": int(1e6),
        "n_parallel_jobs": 3,
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_progress_solve": False,
        "show_summary": False,
        "tolerance_solve": 1e-7,
    }

    # approximate elasticity
    elast_1_step = elasticity_1_step(controls, calibration)

    # store results
    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_" + setup_name + "_elasticity_approx_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(elast_1_step, outfile, ensure_ascii=False, indent=2)
