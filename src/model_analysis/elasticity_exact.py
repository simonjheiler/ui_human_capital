import copy
import json
import src.utilities.istarmap_3_8  # noqa, noreorder
import multiprocessing
import sys

import numpy as np
import tqdm

from bld.project_paths import project_paths_join as ppj
from src.model_analysis.run_utils import _solve_run


#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def elasticity_exact(controls, calibration):

    # load controls
    show_progress = controls["show_progress"]
    n_parallel_jobs = controls["n_parallel_jobs"]
    shock_size = controls["step_size_elasticity"]
    n_simulations = controls["n_simulations"]

    # load variables
    n_periods_working = calibration["n_periods_working"]
    n_periods_retired = calibration["n_periods_retired"]
    n_types = calibration["n_types"]
    type_weights = np.array(calibration["type_weights"])
    ui_replacement_rate_vector = np.array(calibration["ui_replacement_rate_vector"])

    # compute derived variables
    n_years_working = int(n_periods_working / 4)
    n_runs = (n_years_working + 1) * 2  # no shock + up/downward shock in every year

    # initialize objects
    job_finding_rate_searching_all = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )
    share_nonemployed = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )
    share_unemployed_loss = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )
    share_searching = np.full((n_types, n_periods_working, int(n_runs / 2), 2), np.nan)
    wage_hc_factor_pre_displacement = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )
    total_benefits = np.full((n_types, int(n_runs / 2), 2), np.nan)
    pv_government_spending = np.full((n_types, int(n_runs / 2), 2), np.nan)
    net_government_spending_working = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )
    net_government_spending_all = np.full(
        (n_types, n_periods_working + n_periods_retired, int(n_runs / 2), 2), np.nan
    )
    marginal_utility_nonemployed = np.full(
        (n_types, n_periods_working, int(n_runs / 2), 2), np.nan
    )

    # generate shocked input vectors
    shock_vector = np.array([shock_size, -shock_size])
    index_start = np.full(n_years_working, np.nan, dtype=int)
    index_end = np.full(n_years_working, np.nan, dtype=int)
    for year_idx in range(n_years_working):
        period_idx_start = int(year_idx * 4)
        period_idx_end = int(min(period_idx_start + 4, n_periods_working))
        index_start[year_idx] = period_idx_start
        index_end[year_idx] = period_idx_end

    ui_replacement_rate_vector_all = np.repeat(
        ui_replacement_rate_vector, n_runs
    ).reshape((n_types, n_periods_working, n_runs))
    for year_idx in range(n_years_working):
        for shock_idx, shock in enumerate(shock_vector):
            ui_replacement_rate_vector_all[
                :,
                index_start[year_idx] : index_end[year_idx],
                (
                    year_idx + 1 + (n_years_working + 1) * shock_idx
                ),  # run without shock first
            ] += shock

    # define program for parallel computation
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
    with multiprocessing.Pool(n_parallel_jobs) as pool:
        if show_progress:
            out = tuple(
                tqdm.tqdm(
                    pool.istarmap(_solve_run, inputs),
                    total=n_runs,
                    desc="Elasticity",
                    ascii=True,
                    ncols=94,
                )
            )
        else:
            out = pool.starmap(_solve_run, inputs)

    # extract results
    for run_idx in range(int(n_runs / 2)):
        for shock_idx in range(2):
            tmp = out[run_idx + (n_years_working + 1) * shock_idx]
            job_finding_rate_searching_all[:, :, run_idx, shock_idx] = np.array(
                tmp["job_finding_rate_searching_all_mean"]
            )
            marginal_utility_nonemployed[:, :, run_idx, shock_idx] = np.array(
                tmp["marginal_utility_nonemployed_mean"]
            )
            net_government_spending_working[:, :, run_idx, shock_idx] = np.array(
                tmp["net_government_spending_working"]
            )
            net_government_spending_all[:, :, run_idx, shock_idx] = np.array(
                tmp["net_government_spending_all"]
            )
            pv_government_spending[:, run_idx, shock_idx] = np.array(
                tmp["pv_government_spending"]
            )
            share_nonemployed[:, :, run_idx, shock_idx] = np.array(
                tmp["share_nonemployed"]
            )
            share_unemployed_loss[:, :, run_idx, shock_idx] = np.array(
                tmp["share_unemployed_loss"]
            )
            share_searching[:, :, run_idx, shock_idx] = np.array(tmp["share_searching"])
            total_benefits[:, run_idx, shock_idx] = np.array(tmp["total_benefits"])
            wage_hc_factor_pre_displacement[:, :, run_idx, shock_idx] = np.array(
                tmp["wage_hc_factor_pre_displacement_mean"]
            )

    # average over types
    average_ui_replacement_rate_vector = np.average(
        ui_replacement_rate_vector, weights=type_weights, axis=0
    )
    average_job_finding_rate_searching_all = np.average(
        job_finding_rate_searching_all, weights=type_weights, axis=0
    )
    average_marginal_utility_nonemployed = np.average(
        marginal_utility_nonemployed, weights=type_weights, axis=0
    )
    average_net_government_spending_working = np.average(
        net_government_spending_working, weights=type_weights, axis=0
    )
    average_net_government_spending_all = np.average(
        net_government_spending_all, weights=type_weights, axis=0
    )
    average_pv_government_spending = np.average(
        pv_government_spending, weights=type_weights, axis=0
    )
    average_share_nonemployed = np.average(
        share_nonemployed, weights=type_weights, axis=0
    )
    average_share_unemployed_loss = np.average(
        share_unemployed_loss, weights=type_weights, axis=0
    )
    average_share_searching = np.average(share_searching, weights=type_weights, axis=0)
    average_total_benefits = np.average(total_benefits, weights=type_weights, axis=0)
    average_wage_hc_factor_pre_displacement = np.average(
        wage_hc_factor_pre_displacement, weights=type_weights, axis=0
    )

    # calculate elasticities
    average_share_nonemployed_base = average_share_nonemployed[:, 0, 0]
    average_share_searching_base = average_share_searching[:, 0, 0]
    average_wage_hc_factor_pre_displacement_base = (
        average_wage_hc_factor_pre_displacement[:, 0, 0]
    )

    share_nonemployed_base = share_nonemployed[:, :, 0, 0]
    share_searching_base = share_searching[:, :, 0, 0]
    wage_hc_factor_pre_displacement_base = wage_hc_factor_pre_displacement[:, :, 0, 0]

    # Computing means of variables by groups of age
    average_ui_replacement_rate_yearly = np.full(n_years_working, np.nan)

    average_job_finding_rate_yearly_base = np.full(n_years_working, np.nan)
    average_share_nonemployed_yearly_base = np.full(n_years_working, np.nan)
    average_share_unemployed_loss_yearly_base = np.full(n_years_working, np.nan)
    average_share_searching_yearly_base = np.full(n_years_working, np.nan)

    average_job_finding_rate_yearly_shocked = np.full((n_years_working, 2), np.nan)
    average_share_nonemployed_yearly_shocked = np.full((n_years_working, 2), np.nan)
    average_share_unemployed_loss_yearly_shocked = np.full((n_years_working, 2), np.nan)
    average_share_searching_yearly_shocked = np.full((n_years_working, 2), np.nan)

    ui_replacement_rate_yearly = np.full((n_types, n_years_working), np.nan)

    job_finding_rate_yearly_base = np.full((n_types, n_years_working), np.nan)
    share_nonemployed_yearly_base = np.full((n_types, n_years_working), np.nan)
    share_unemployed_loss_yearly_base = np.full((n_types, n_years_working), np.nan)
    share_searching_yearly_base = np.full((n_types, n_years_working), np.nan)

    job_finding_rate_yearly_shocked = np.full((n_types, n_years_working, 2), np.nan)
    share_nonemployed_yearly_shocked = np.full((n_types, n_years_working, 2), np.nan)
    share_unemployed_loss_yearly_shocked = np.full(
        (n_types, n_years_working, 2), np.nan
    )
    share_searching_yearly_shocked = np.full((n_types, n_years_working, 2), np.nan)

    for i in range(n_years_working):

        idx_start = index_start[i]
        idx_end = index_end[i]

        average_ui_replacement_rate_yearly[i] = np.mean(
            average_ui_replacement_rate_vector[idx_start:idx_end]
        )

        average_job_finding_rate_yearly_base[i] = np.mean(
            average_job_finding_rate_searching_all[idx_start:idx_end, 0, 0]
        )
        average_share_nonemployed_yearly_base[i] = np.mean(
            average_share_nonemployed[idx_start:idx_end, 0, 0]
        )
        average_share_unemployed_loss_yearly_base[i] = np.mean(
            average_share_unemployed_loss[idx_start:idx_end, 0, 0]
        )
        average_share_searching_yearly_base[i] = np.mean(
            average_share_searching[idx_start:idx_end, 0, 0]
        )

        for z in range(2):
            average_job_finding_rate_yearly_shocked[i, z] = np.mean(
                average_job_finding_rate_searching_all[idx_start:idx_end, i + 1, z]
            )
            average_share_nonemployed_yearly_shocked[i, z] = np.mean(
                average_share_nonemployed[idx_start:idx_end, i + 1, z]
            )
            average_share_unemployed_loss_yearly_shocked[i, z] = np.mean(
                average_share_unemployed_loss[idx_start:idx_end, i + 1, z]
            )
            average_share_searching_yearly_shocked[i, z] = np.mean(
                average_share_searching[idx_start:idx_end, i + 1, z]
            )

        ui_replacement_rate_yearly[:, i] = np.mean(
            ui_replacement_rate_vector[:, idx_start:idx_end], axis=1
        )

        job_finding_rate_yearly_base[:, i] = np.mean(
            job_finding_rate_searching_all[:, idx_start:idx_end, 0, 0], axis=1
        )
        share_nonemployed_yearly_base[:, i] = np.mean(
            share_nonemployed[:, idx_start:idx_end, 0, 0], axis=1
        )
        share_unemployed_loss_yearly_base[:, i] = np.mean(
            share_unemployed_loss[:, idx_start:idx_end, 0, 0], axis=1
        )
        share_searching_yearly_base[:, i] = np.mean(
            share_searching[:, idx_start:idx_end, 0, 0], axis=1
        )

        for z in range(2):
            job_finding_rate_yearly_shocked[:, i, z] = np.mean(
                job_finding_rate_searching_all[:, idx_start:idx_end, i + 1, z], axis=1
            )
            share_nonemployed_yearly_shocked[:, i, z] = np.mean(
                share_nonemployed[:, idx_start:idx_end, i + 1, z], axis=1
            )
            share_unemployed_loss_yearly_shocked[:, i, z] = np.mean(
                share_unemployed_loss[:, idx_start:idx_end, i + 1, z], axis=1
            )
            share_searching_yearly_shocked[:, i, z] = np.mean(
                share_searching[:, idx_start:idx_end, i + 1, z], axis=1
            )

    # Computing elasticities and cross elasticities
    delta_average_unemployment = (
        average_share_nonemployed[:, :, 0] - average_share_nonemployed[:, :, 1]
    )
    delta_average_job_finding_rate_yearly = (
        average_job_finding_rate_yearly_shocked[:, 0]
        - average_job_finding_rate_yearly_shocked[:, 1]
    )
    delta_average_share_nonemployed_yearly = (
        average_share_nonemployed_yearly_shocked[:, 0]
        - average_share_nonemployed_yearly_shocked[:, 1]
    )
    delta_average_share_unemployed_loss_yearly = (
        average_share_unemployed_loss_yearly_shocked[:, 0]
        - average_share_unemployed_loss_yearly_shocked[:, 1]
    )
    delta_average_share_searching_yearly = (
        average_share_searching_yearly_shocked[:, 0]
        - average_share_searching_yearly_shocked[:, 1]
    )

    delta_unemployment = share_nonemployed[:, :, :, 0] - share_nonemployed[:, :, :, 1]
    delta_job_finding_rate_yearly = (
        job_finding_rate_yearly_shocked[:, :, 0]
        - job_finding_rate_yearly_shocked[:, :, 1]
    )
    delta_share_nonemployed_yearly = (
        share_nonemployed_yearly_shocked[:, :, 0]
        - share_nonemployed_yearly_shocked[:, :, 1]
    )
    delta_share_unemployed_loss_yearly = (
        share_unemployed_loss_yearly_shocked[:, :, 0]
        - share_unemployed_loss_yearly_shocked[:, :, 1]
    )
    delta_share_searching_yearly = (
        share_searching_yearly_shocked[:, :, 0]
        - share_searching_yearly_shocked[:, :, 1]
    )

    average_elasticity_job_finding_rate_yearly = (
        delta_average_job_finding_rate_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * average_job_finding_rate_yearly_base)
    average_elasticity_share_nonemployed_yearly = (
        delta_average_share_nonemployed_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * average_share_nonemployed_yearly_base)
    average_elasticity_share_unemployed_loss_yearly = (
        delta_average_share_unemployed_loss_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * average_share_unemployed_loss_yearly_base)
    average_elasticity_share_searching_yearly = (
        delta_average_share_searching_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * average_share_searching_yearly_base)

    elasticity_job_finding_rate_yearly = (
        delta_job_finding_rate_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * job_finding_rate_yearly_base)
    elasticity_share_nonemployed_yearly = (
        delta_share_nonemployed_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * share_nonemployed_yearly_base)
    elasticity_share_unemployed_loss_yearly = (
        delta_share_unemployed_loss_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * share_unemployed_loss_yearly_base)
    elasticity_share_searching_yearly = (
        delta_share_searching_yearly * ui_replacement_rate_yearly
    ) / (2 * shock_size * share_searching_yearly_base)

    # more precise computation of elasticity of unemployment
    average_elasticity_unemployment = (
        delta_average_unemployment
        * np.repeat(average_ui_replacement_rate_vector, (n_years_working + 1)).reshape(
            (n_periods_working, (n_years_working + 1))
        )
    ) / (
        2
        * shock_size
        * np.repeat(average_share_nonemployed[:, 0, 0], (n_years_working + 1)).reshape(
            (n_periods_working, (n_years_working + 1))
        )
    )
    average_elasticity_unemployment_mean = np.full(n_years_working, np.nan)
    for year_idx in range(n_years_working):
        average_elasticity_unemployment_mean[year_idx] = np.sum(
            average_elasticity_unemployment[
                index_start[year_idx] : index_end[year_idx], year_idx + 1
            ]
            * average_share_nonemployed[
                index_start[year_idx] : index_end[year_idx], 0, 0
            ]
        ) / np.sum(
            average_share_nonemployed[index_start[year_idx] : index_end[year_idx], 0, 0]
        )

    elasticity_unemployment = (
        delta_unemployment
        * np.repeat(ui_replacement_rate_vector, (n_years_working + 1)).reshape(
            (n_types, n_periods_working, (n_years_working + 1))
        )
    ) / (
        2
        * shock_size
        * np.repeat(share_nonemployed[:, :, 0, 0], (n_years_working + 1)).reshape(
            (n_types, n_periods_working, (n_years_working + 1))
        )
    )
    elasticity_unemployment_mean = np.full((n_types, n_years_working), np.nan)
    for year_idx in range(n_years_working):
        elasticity_unemployment_mean[:, year_idx] = np.sum(
            elasticity_unemployment[
                :, index_start[year_idx] : index_end[year_idx], year_idx + 1
            ]
            * share_nonemployed[:, index_start[year_idx] : index_end[year_idx], 0, 0],
            axis=1,
        ) / np.sum(
            share_nonemployed[:, index_start[year_idx] : index_end[year_idx], 0, 0],
            axis=1,
        )

    # cross elasticities for j age groups
    # simulation
    wage_level = calibration["wage_level"]
    discount_factor = calibration["discount_factor"]
    discount_factor_compounded_vector = discount_factor ** np.linspace(
        0, 179, n_periods_working
    )

    average_cross_elasticity_benefits_yearly = np.full(n_years_working, np.nan)
    average_cross_elasticity_benefits_discounted_yearly = np.full(
        n_years_working, np.nan
    )
    average_cross_elasticity_total_benefits_yearly = np.full(n_years_working, np.nan)
    average_cross_elasticity_pv_government_spending_yearly = np.full(
        n_years_working, np.nan
    )
    average_cross_elasticity_net_government_spending_working_yearly = np.full(
        n_years_working, np.nan
    )
    average_cross_elasticity_net_government_spending_all_yearly = np.full(
        n_years_working, np.nan
    )

    average_adjustment_factor = np.full(n_years_working, np.nan)
    average_adjustment_factor_discounted = np.full(n_years_working, np.nan)

    average_marginal_utility_nonemployed_yearly = np.full(
        (n_years_working, n_years_working + 1), np.nan
    )

    cross_elasticity_benefits_yearly = np.full((n_types, n_years_working), np.nan)
    cross_elasticity_benefits_discounted_yearly = np.full(
        (n_types, n_years_working), np.nan
    )
    cross_elasticity_total_benefits_yearly = np.full((n_types, n_years_working), np.nan)
    cross_elasticity_pv_government_spending_yearly = np.full(
        (n_types, n_years_working), np.nan
    )
    cross_elasticity_net_government_spending_working_yearly = np.full(
        (n_types, n_years_working), np.nan
    )
    cross_elasticity_net_government_spending_all_yearly = np.full(
        (n_types, n_years_working), np.nan
    )

    adjustment_factor = np.full((n_types, n_years_working), np.nan)
    adjustment_factor_discounted = np.full((n_types, n_years_working), np.nan)

    marginal_utility_nonemployed_yearly = np.full(
        (n_types, n_years_working, n_years_working + 1), np.nan
    )

    for i in range(n_years_working):

        idx_start = index_start[i]
        idx_end = index_end[i]

        average_cross_elasticity_benefits_yearly[i] = np.sum(
            (
                average_share_nonemployed[:, i + 1, 0]
                - average_share_nonemployed[:, i + 1, 1]
            )
            * average_ui_replacement_rate_vector
            * wage_level
            * average_wage_hc_factor_pre_displacement_base
        ) / np.sum(
            average_share_nonemployed_base[idx_start:idx_end]
            * 2
            * shock_size
            * wage_level
            * average_wage_hc_factor_pre_displacement_base[idx_start:idx_end],
        )
        average_cross_elasticity_benefits_discounted_yearly[i] = np.sum(
            discount_factor_compounded_vector
            * (
                average_share_nonemployed[:, i + 1, 0]
                - average_share_nonemployed[:, i + 1, 1]
            )
            * average_ui_replacement_rate_vector
            * average_wage_hc_factor_pre_displacement_base
            * wage_level
        ) / np.sum(
            discount_factor_compounded_vector[idx_start:idx_end]
            * average_share_nonemployed_base[idx_start:idx_end]
            * 2
            * shock_size
            * wage_level
            * average_wage_hc_factor_pre_displacement_base[idx_start:idx_end]
        )
        average_cross_elasticity_total_benefits_yearly[i] = (
            average_total_benefits[i + 1, 0] - average_total_benefits[i + 1, 1]
        ) / 2
        average_cross_elasticity_pv_government_spending_yearly[i] = (
            average_pv_government_spending[i + 1, 0]
            - average_pv_government_spending[i + 1, 1]
        ) / 2
        average_cross_elasticity_net_government_spending_working_yearly[i] = (
            np.sum(
                average_net_government_spending_working[:, i + 1, 0]
                - average_net_government_spending_working[:, i + 1, 1]
            )
            / 2
        )
        average_cross_elasticity_net_government_spending_all_yearly[i] = (
            np.sum(
                average_net_government_spending_all[:, i + 1, 0]
                - average_net_government_spending_all[:, i + 1, 1]
            )
            / 2
        )
        average_adjustment_factor[i] = np.sum(
            shock_size
            * wage_level
            * average_wage_hc_factor_pre_displacement_base[idx_start:idx_end]
            * average_share_nonemployed_base[idx_start:idx_end]
        )
        average_adjustment_factor_discounted[i] = np.sum(
            shock_size
            * wage_level
            * discount_factor_compounded_vector[idx_start:idx_end]
            * average_wage_hc_factor_pre_displacement_base[idx_start:idx_end]
            * average_share_nonemployed_base[idx_start:idx_end]
        )
        average_marginal_utility_nonemployed_yearly[i, :] = np.mean(
            average_marginal_utility_nonemployed[idx_start:idx_end, :, 0]
        )

        cross_elasticity_benefits_yearly[:, i] = np.sum(
            (share_nonemployed[:, :, i + 1, 0] - share_nonemployed[:, :, i + 1, 1])
            * ui_replacement_rate_vector
            * wage_level
            * wage_hc_factor_pre_displacement_base,
            axis=1,
        ) / np.sum(
            share_nonemployed_base[:, idx_start:idx_end]
            * 2
            * shock_size
            * wage_level
            * wage_hc_factor_pre_displacement_base[:, idx_start:idx_end],
            axis=1,
        )
        cross_elasticity_benefits_discounted_yearly[:, i] = np.sum(
            discount_factor_compounded_vector
            * (share_nonemployed[:, :, i + 1, 0] - share_nonemployed[:, :, i + 1, 1])
            * ui_replacement_rate_vector
            * wage_hc_factor_pre_displacement_base
            * wage_level,
            axis=1,
        ) / np.sum(
            discount_factor_compounded_vector[idx_start:idx_end]
            * share_nonemployed_base[:, idx_start:idx_end]
            * 2
            * shock_size
            * wage_level
            * wage_hc_factor_pre_displacement_base[:, idx_start:idx_end],
            axis=1,
        )
        cross_elasticity_total_benefits_yearly[:, i] = (
            total_benefits[:, i + 1, 0] - total_benefits[:, i + 1, 1]
        ) / 2
        cross_elasticity_pv_government_spending_yearly[:, i] = (
            pv_government_spending[:, i + 1, 0] - pv_government_spending[:, i + 1, 1]
        ) / 2
        cross_elasticity_net_government_spending_working_yearly[:, i] = (
            np.sum(
                net_government_spending_working[:, :, i + 1, 0]
                - net_government_spending_working[:, :, i + 1, 1]
            )
            / 2
        )
        cross_elasticity_net_government_spending_all_yearly[:, i] = (
            np.sum(
                net_government_spending_all[:, :, i + 1, 0]
                - net_government_spending_all[:, :, i + 1, 1]
            )
            / 2
        )
        adjustment_factor[:, i] = np.sum(
            shock_size
            * wage_level
            * wage_hc_factor_pre_displacement_base[:, idx_start:idx_end]
            * share_nonemployed_base[:, idx_start:idx_end]
        )
        adjustment_factor_discounted[:, i] = np.sum(
            shock_size
            * wage_level
            * discount_factor_compounded_vector[idx_start:idx_end]
            * wage_hc_factor_pre_displacement_base[:, idx_start:idx_end]
            * share_nonemployed_base[:, idx_start:idx_end]
        )
        marginal_utility_nonemployed_yearly[:, i, :] = np.mean(
            marginal_utility_nonemployed[:, idx_start:idx_end, :, 0], axis=1
        )

    average_cross_elasticity_total_benefits_yearly = (
        average_cross_elasticity_total_benefits_yearly
        / (average_adjustment_factor * n_simulations)
        - 1
    )
    average_cross_elasticity_pv_government_spending_discounted_yearly = (
        -average_cross_elasticity_pv_government_spending_yearly
        / (average_adjustment_factor_discounted * n_simulations)
        - 1
    )
    average_cross_elasticity_pv_government_spending_yearly = (
        -average_cross_elasticity_pv_government_spending_yearly
        / (average_adjustment_factor * n_simulations)
        - 1
    )
    average_cross_elasticity_net_government_spending_working_yearly = (
        -average_cross_elasticity_net_government_spending_working_yearly
        / (average_adjustment_factor * n_simulations)
        - 1
    )
    average_cross_elasticity_net_government_spending_all_yearly = (
        -average_cross_elasticity_net_government_spending_all_yearly
        / (average_adjustment_factor * n_simulations)
        - 1
    )

    cross_elasticity_total_benefits_yearly = (
        cross_elasticity_total_benefits_yearly / (adjustment_factor * n_simulations) - 1
    )
    cross_elasticity_pv_government_spending_discounted_yearly = (
        -cross_elasticity_pv_government_spending_yearly
        / (adjustment_factor_discounted * n_simulations)
        - 1
    )
    cross_elasticity_pv_government_spending_yearly = (
        -cross_elasticity_pv_government_spending_yearly
        / (adjustment_factor * n_simulations)
        - 1
    )
    cross_elasticity_net_government_spending_working_yearly = (
        -cross_elasticity_net_government_spending_working_yearly
        / (adjustment_factor * n_simulations)
        - 1
    )
    cross_elasticity_net_government_spending_all_yearly = (
        -cross_elasticity_net_government_spending_all_yearly
        / (adjustment_factor * n_simulations)
        - 1
    )

    average_cross_elasticity_total_benefits_mean = np.mean(
        average_cross_elasticity_total_benefits_yearly
    )
    average_cross_elasticity_benefits_mean = np.mean(
        average_cross_elasticity_benefits_yearly
    )
    average_cross_elasticity_pv_government_spending_mean = np.mean(
        average_cross_elasticity_pv_government_spending_yearly
    )
    average_cross_elasticity_net_government_spending_working_mean = np.mean(
        average_cross_elasticity_net_government_spending_working_yearly
    )
    average_elasticity_share_nonemployed_mean = np.mean(
        average_elasticity_share_nonemployed_yearly
    )
    average_elasticity_share_nonemployed_std = np.std(
        average_elasticity_share_nonemployed_yearly
    )
    average_elasticity_job_finding_rate_std = np.std(
        average_elasticity_job_finding_rate_yearly
    )
    average_elasticity_share_searching_std = np.std(
        average_elasticity_share_searching_yearly
    )
    average_cross_elasticity_benefits_std = np.std(
        average_cross_elasticity_benefits_yearly
    )
    average_cross_elasticity_total_benefits_std = np.std(
        average_cross_elasticity_total_benefits_yearly
    )
    average_cross_elasticity_pv_government_spending_std = np.std(
        average_cross_elasticity_pv_government_spending_yearly
    )
    average_cross_elasticity_net_government_spending_working_std = np.std(
        average_cross_elasticity_net_government_spending_working_yearly
    )

    cross_elasticity_total_benefits_mean = np.mean(
        cross_elasticity_total_benefits_yearly, axis=1
    )
    cross_elasticity_benefits_mean = np.mean(cross_elasticity_benefits_yearly, axis=1)
    cross_elasticity_pv_government_spending_mean = np.mean(
        cross_elasticity_pv_government_spending_yearly, axis=1
    )
    cross_elasticity_net_government_spending_working_mean = np.mean(
        cross_elasticity_net_government_spending_working_yearly, axis=1
    )
    elasticity_share_nonemployed_mean = np.mean(
        elasticity_share_nonemployed_yearly, axis=1
    )
    elasticity_share_nonemployed_std = np.std(
        elasticity_share_nonemployed_yearly, axis=1
    )
    elasticity_job_finding_rate_std = np.std(elasticity_job_finding_rate_yearly, axis=1)
    elasticity_share_searching_std = np.std(elasticity_share_searching_yearly, axis=1)
    cross_elasticity_benefits_std = np.std(cross_elasticity_benefits_yearly, axis=1)
    cross_elasticity_total_benefits_std = np.std(
        cross_elasticity_total_benefits_yearly, axis=1
    )
    cross_elasticity_pv_government_spending_std = np.std(
        cross_elasticity_pv_government_spending_yearly, axis=1
    )
    cross_elasticity_net_government_spending_working_std = np.std(
        cross_elasticity_net_government_spending_working_yearly, axis=1
    )

    # store results
    out = {
        "average_cross_elasticity_benefits_discounted_yearly": average_cross_elasticity_benefits_discounted_yearly,  # noqa:B950
        "average_cross_elasticity_benefits_mean": average_cross_elasticity_benefits_mean,
        "average_cross_elasticity_benefits_std": average_cross_elasticity_benefits_std,
        "average_cross_elasticity_benefits_yearly": average_cross_elasticity_benefits_yearly,
        "average_cross_elasticity_net_government_spending_all_yearly": average_cross_elasticity_net_government_spending_all_yearly,  # noqa:B950
        "average_cross_elasticity_net_government_spending_working_mean": average_cross_elasticity_net_government_spending_working_mean,  # noqa:B950
        "average_cross_elasticity_net_government_spending_working_std": average_cross_elasticity_net_government_spending_working_std,  # noqa:B950
        "average_cross_elasticity_net_government_spending_working_yearly": average_cross_elasticity_net_government_spending_working_yearly,  # noqa:B950
        "average_cross_elasticity_pv_government_spending_discounted_yearly": average_cross_elasticity_pv_government_spending_discounted_yearly,  # noqa:B950
        "average_cross_elasticity_pv_government_spending_mean": average_cross_elasticity_pv_government_spending_mean,  # noqa:B950
        "average_cross_elasticity_pv_government_spending_std": average_cross_elasticity_pv_government_spending_std,  # noqa:B950
        "average_cross_elasticity_pv_government_spending_yearly": average_cross_elasticity_pv_government_spending_yearly,  # noqa:B950
        "average_cross_elasticity_total_benefits_mean": average_cross_elasticity_total_benefits_mean,  # noqa:B950
        "average_cross_elasticity_total_benefits_std": average_cross_elasticity_total_benefits_std,  # noqa:B950
        "average_cross_elasticity_total_benefits_yearly": average_cross_elasticity_total_benefits_yearly,  # noqa:B950
        "average_adjustment_factor": average_adjustment_factor,
        "average_adjustment_factor_discounted": average_adjustment_factor_discounted,
        "average_elasticity_unemployment_mean": average_elasticity_unemployment_mean,
        "average_elasticity_job_finding_rate_yearly": average_elasticity_job_finding_rate_yearly,  # noqa:B950
        "average_elasticity_share_nonemployed_yearly": average_elasticity_share_nonemployed_yearly,  # noqa:B950
        "average_elasticity_share_unemployed_loss_yearly": average_elasticity_share_unemployed_loss_yearly,  # noqa:B950
        "average_elasticity_share_searching_yearly": average_elasticity_share_searching_yearly,
        "average_elasticity_share_nonemployed_mean": average_elasticity_share_nonemployed_mean,
        "average_elasticity_share_nonemployed_std": average_elasticity_share_nonemployed_std,
        "average_elasticity_job_finding_rate_std": average_elasticity_job_finding_rate_std,
        "average_elasticity_share_searching_std": average_elasticity_share_searching_std,
        "average_job_finding_rate_searching_all": average_job_finding_rate_searching_all,
        "average_job_finding_rate_yearly_base": average_job_finding_rate_yearly_base,
        "average_job_finding_rate_yearly_shocked": average_job_finding_rate_yearly_shocked,
        "average_marginal_utility_nonemployed_yearly": average_marginal_utility_nonemployed_yearly,  # noqa:B950
        "average_marginal_utility_nonemployed": average_marginal_utility_nonemployed,
        "average_net_government_spending_all": average_net_government_spending_all,
        "average_net_government_spending_working": average_net_government_spending_working,
        "average_pv_government_spending": average_pv_government_spending,
        "average_share_nonemployed": average_share_nonemployed,
        "average_share_nonemployed_base": average_share_nonemployed_base,
        "average_share_nonemployed_yearly_base": average_share_nonemployed_yearly_base,
        "average_share_nonemployed_yearly_shocked": average_share_nonemployed_yearly_shocked,
        "average_share_searching": average_share_searching,
        "average_share_searching_base": average_share_searching_base,
        "average_share_searching_yearly_base": average_share_searching_yearly_base,
        "average_share_searching_yearly_shocked": average_share_searching_yearly_shocked,
        "average_share_unemployed_loss": average_share_unemployed_loss,
        "average_share_unemployed_loss_yearly_base": average_share_unemployed_loss_yearly_base,
        "average_share_unemployed_loss_yearly_shocked": average_share_unemployed_loss_yearly_shocked,  # noqa:B950
        "average_total_benefits": average_total_benefits,
        "average_wage_hc_factor_pre_displacement": average_wage_hc_factor_pre_displacement,
        "average_wage_hc_factor_pre_displacement_base": average_wage_hc_factor_pre_displacement_base,  # noqa:B950
        "cross_elasticity_benefits_discounted_yearly": cross_elasticity_benefits_discounted_yearly,  # noqa:B950
        "cross_elasticity_benefits_mean": cross_elasticity_benefits_mean,
        "cross_elasticity_benefits_std": cross_elasticity_benefits_std,
        "cross_elasticity_benefits_yearly": cross_elasticity_benefits_yearly,
        "cross_elasticity_net_government_spending_all_yearly": cross_elasticity_net_government_spending_all_yearly,  # noqa:B950
        "cross_elasticity_net_government_spending_working_mean": cross_elasticity_net_government_spending_working_mean,  # noqa:B950
        "cross_elasticity_net_government_spending_working_std": cross_elasticity_net_government_spending_working_std,  # noqa:B950
        "cross_elasticity_net_government_spending_working_yearly": cross_elasticity_net_government_spending_working_yearly,  # noqa:B950
        "cross_elasticity_pv_government_spending_discounted_yearly": cross_elasticity_pv_government_spending_discounted_yearly,  # noqa:B950
        "cross_elasticity_pv_government_spending_mean": cross_elasticity_pv_government_spending_mean,  # noqa:B950
        "cross_elasticity_pv_government_spending_std": cross_elasticity_pv_government_spending_std,  # noqa:B950
        "cross_elasticity_pv_government_spending_yearly": cross_elasticity_pv_government_spending_yearly,  # noqa:B950
        "cross_elasticity_total_benefits_mean": cross_elasticity_total_benefits_mean,
        "cross_elasticity_total_benefits_std": cross_elasticity_total_benefits_std,
        "cross_elasticity_total_benefits_yearly": cross_elasticity_total_benefits_yearly,
        "adjustment_factor": adjustment_factor,
        "adjustment_factor_discounted": adjustment_factor_discounted,
        "elasticity_unemployment_mean": elasticity_unemployment_mean,
        "elasticity_job_finding_rate_yearly": elasticity_job_finding_rate_yearly,
        "elasticity_share_nonemployed_yearly": elasticity_share_nonemployed_yearly,
        "elasticity_share_unemployed_loss_yearly": elasticity_share_unemployed_loss_yearly,
        "elasticity_share_searching_yearly": elasticity_share_searching_yearly,
        "elasticity_share_nonemployed_mean": elasticity_share_nonemployed_mean,
        "elasticity_share_nonemployed_std": elasticity_share_nonemployed_std,
        "elasticity_job_finding_rate_std": elasticity_job_finding_rate_std,
        "elasticity_share_searching_std": elasticity_share_searching_std,
        "job_finding_rate_searching_all": job_finding_rate_searching_all,
        "job_finding_rate_yearly_base": job_finding_rate_yearly_base,
        "job_finding_rate_yearly_shocked": job_finding_rate_yearly_shocked,
        "marginal_utility_nonemployed_yearly": marginal_utility_nonemployed_yearly,
        "marginal_utility_nonemployed": marginal_utility_nonemployed,
        "net_government_spending_all": net_government_spending_all,
        "net_government_spending_working": net_government_spending_working,
        "pv_government_spending": pv_government_spending,
        "share_nonemployed": share_nonemployed,
        "share_nonemployed_base": share_nonemployed_base,
        "share_nonemployed_yearly_base": share_nonemployed_yearly_base,
        "share_nonemployed_yearly_shocked": share_nonemployed_yearly_shocked,
        "share_searching": share_searching,
        "share_searching_base": share_searching_base,
        "share_searching_yearly_base": share_searching_yearly_base,
        "share_searching_yearly_shocked": share_searching_yearly_shocked,
        "share_unemployed_loss": share_unemployed_loss,
        "share_unemployed_loss_yearly_base": share_unemployed_loss_yearly_base,
        "share_unemployed_loss_yearly_shocked": share_unemployed_loss_yearly_shocked,
        "shock_size": shock_size,
        "shock_vector": shock_vector,
        "total_benefits": total_benefits,
        "ui_replacement_rate_vector_all": ui_replacement_rate_vector_all,
        "wage_hc_factor_pre_displacement": wage_hc_factor_pre_displacement,
        "wage_hc_factor_pre_displacement_base": wage_hc_factor_pre_displacement_base,
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
        setup_name = "base_test_1"
        method = "linear"

    # load calibration
    calibration = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_solve_max": 1,
        "n_parallel_jobs": 12,
        "n_simulations": int(1e6),
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_progress": True,
        "show_progress_solve": False,
        "show_summary": False,
        "step_size_elasticity": 0.01,
        "tolerance_solve": 1e-7,
    }

    # compute elasticity
    elast_exact = elasticity_exact(controls, calibration)

    # store results
    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_" + setup_name + "_elasticity_exact_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(elast_exact, outfile, ensure_ascii=False, indent=2)
