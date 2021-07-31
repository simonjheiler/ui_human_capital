""" Compute consumption equivalents.

This module computes the equivalent consumption changes required
to compensate for welfare differentials.

"""
#####################################################
# IMPORTS
#####################################################
import json
import sys

import numba as nb
import numpy as np
import pandas as pd
from scipy import interpolate

from bld.project_paths import project_paths_join as ppj
from src.utilities.interpolation_utils import interpolate_2d_ordered_to_unordered

#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


@nb.njit
def consumption_utility(x):
    if risk_aversion_coefficient == 1:
        return np.log(x)
    else:
        return x ** (1 - risk_aversion_coefficient) / (1 - risk_aversion_coefficient)


def _hc_after_loss_n_agents(
    hc_before_loss,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):

    hc_after_loss = np.full(hc_before_loss.shape, np.nan)

    for type_idx in range(n_types):
        hc_after_loss[type_idx, ...] = _hc_after_loss_1_agent(
            hc_before_loss[type_idx, ...],
            wage_loss_factor_vector[type_idx, :],
            wage_loss_reference_vector[type_idx, :],
            period_idx,
        )

    return hc_after_loss


def _hc_after_loss_1_agent(
    hc_before_loss,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):
    func = interpolate.interp1d(
        wage_loss_reference_vector,
        hc_grid,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    val = np.maximum(
        wage_hc_factor_interpolated_1_agent(
            np.minimum(hc_before_loss, hc_max), wage_loss_reference_vector
        )
        * wage_loss_factor_vector[period_idx],
        wage_hc_factor_interpolated_1_agent(0, wage_loss_reference_vector),
    )

    return func(val)


@nb.njit
def job_finding_probability(x):
    return contact_rate * x


def simulate_ui_benefits(
    pre_displacement_wage, replacement_rate_vector, floor, cap, period_idx
):

    benefits = np.full(pre_displacement_wage.shape, np.nan)
    for type_idx in range(pre_displacement_wage.shape[0]):
        benefits[type_idx, :] = _ui_benefits(
            pre_displacement_wage[type_idx, ...],
            replacement_rate_vector[type_idx, ...],
            floor,
            cap,
            period_idx,
        )

    return benefits


def _ui_benefits(
    pre_displacement_wage,
    replacement_rate_vector,
    floor,
    cap,
    period_idx,
):

    benefits = replacement_rate_vector[..., period_idx] * pre_displacement_wage

    benefits = np.minimum(cap, benefits)
    benefits = np.maximum(floor, benefits)

    return benefits


def wage_hc_factor_interpolated_1_agent(x, wage_hc_factor_vector):
    return interpolate.interp1d(
        hc_grid,
        wage_hc_factor_vector,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(x)


def _get_pv_consumption_utility(calibration, results, controls):

    global contact_rate
    global hc_grid
    global interpolation_method
    global risk_aversion_coefficient
    global n_types
    global hc_max

    # load controls
    interpolation_method = controls["interpolation_method"]

    # load calibration
    assets_grid = np.array(calibration["assets_grid"])
    # assets_max = calibration["assets_max"]
    assets_min = calibration["assets_min"]
    contact_rate = calibration["contact_rate"]
    discount_factor = calibration["discount_factor"]
    n_periods_retired = calibration["n_periods_retired"]
    n_periods_working = calibration["n_periods_working"]
    n_types = calibration["n_types"]
    hc_grid_reduced = np.array(calibration["hc_grid_reduced"])
    hc_loss_probability = np.array(calibration["hc_loss_probability"])
    risk_aversion_coefficient = calibration["risk_aversion_coefficient"]
    separation_rate_vector = np.array(calibration["separation_rate_vector"])
    type_weights = np.array(calibration["type_weights"])
    ui_cap = calibration["ui_cap"]
    ui_floor = calibration["ui_floor"]
    ui_replacement_rate_vector = np.array(calibration["ui_replacement_rate_vector"])
    wage_hc_factor_vector = np.array(calibration["wage_hc_factor_vector"])
    wage_level = calibration["wage_level"]
    wage_loss_factor_vector = np.array(calibration["wage_loss_factor_vector"])
    wage_loss_reference_vector = np.array(calibration["wage_loss_reference_vector"])

    # load results
    policy_consumption_employed = np.array(results["policy_consumption_employed"])
    policy_consumption_unemployed = np.array(results["policy_consumption_unemployed"])
    policy_consumption_unemployed_loss = np.array(
        results["policy_consumption_unemployed_loss"]
    )
    policy_effort_searching = np.array(results["policy_effort_searching"])
    policy_effort_searching_loss = np.array(results["policy_effort_searching_loss"])
    income_tax_rate = np.array(results["equilibrium_instrument_rate"])

    # compute derived parameters
    hc_grid = np.arange(n_periods_working + 1)
    hc_max = np.amax(hc_grid)

    income_tax_rate_vector = np.repeat(income_tax_rate, n_periods_working).reshape(
        (n_types, n_periods_working)
    )
    assets_grid_size = len(assets_grid)
    hc_grid_reduced_size = len(hc_grid_reduced)
    interest_rate = (1 - discount_factor) / discount_factor
    borrowing_limit_n_e_a = np.full(
        (n_types, hc_grid_reduced_size, assets_grid_size), assets_min
    )

    assets_grid_e_a = (
        np.repeat(assets_grid, hc_grid_reduced_size)
        .reshape(assets_grid_size, hc_grid_reduced_size)
        .T
    )
    # assets_grid_e_a1 = np.append(
    #     assets_grid_e_a, np.full((hc_grid_reduced_size, 1), assets_max), axis=1
    # )
    assets_grid_n_e_a = np.tile(assets_grid, n_types * hc_grid_reduced_size).reshape(
        (n_types, hc_grid_reduced_size, assets_grid_size)
    )

    hc_grid_reduced_e_a = np.repeat(hc_grid_reduced, assets_grid_size).reshape(
        hc_grid_reduced_size, assets_grid_size
    )
    # hc_grid_reduced_e_a1 = np.append(
    #     hc_grid_reduced_e_a, hc_grid_reduced[..., np.newaxis], axis=1
    # )

    wage_hc_factor_vector_reduced = np.array(
        [wage_hc_factor_vector[i, hc_grid_reduced] for i in range(n_types)]
    )
    wage_hc_factor_grid_n_e_a = np.repeat(
        wage_hc_factor_vector_reduced,
        assets_grid_size,
    ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))

    if ui_cap == "None":
        ui_cap = np.Inf
    if ui_floor == "None":
        ui_floor = 0.0

    # initiate
    period_idx = n_periods_working - 1

    discount_factor_retirement = (1 - discount_factor ** n_periods_retired) / (
        1 - discount_factor
    )
    pv_consumption_utility_employed_next = (
        discount_factor_retirement
        * consumption_utility(policy_consumption_employed[:, :, :, period_idx + 1])
    )
    pv_consumption_utility_unemployed_next = (
        discount_factor_retirement
        * consumption_utility(policy_consumption_unemployed[:, :, :, period_idx + 1])
    )
    pv_consumption_utility_unemployed_loss_next = (
        discount_factor_retirement
        * consumption_utility(
            policy_consumption_unemployed_loss[:, :, :, period_idx + 1]
        )
    )

    while period_idx >= 0:

        # unemployed with human capital loss
        assets_unemployed_loss_next = np.maximum(
            (1 + interest_rate) * assets_grid_n_e_a
            + simulate_ui_benefits(
                wage_level * wage_hc_factor_grid_n_e_a,
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            )
            - policy_consumption_unemployed_loss[:, :, :, period_idx],
            borrowing_limit_n_e_a,
        )

        effort_searching_loss_next = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_employed_loss = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_unemployed_loss = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        for type_idx in range(n_types):
            effort_searching_loss_next[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                policy_effort_searching_loss[type_idx, :, :, period_idx + 1],
                hc_grid_reduced_e_a,
                assets_unemployed_loss_next[type_idx, :, :],
                interpolation_method,
            )

            continuation_value_employed_loss[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_employed_next[type_idx, :, :],
                _hc_after_loss_1_agent(
                    hc_grid_reduced_e_a,
                    wage_loss_factor_vector[type_idx, :],
                    wage_loss_reference_vector[type_idx, :],
                    period_idx + 1,
                ),
                assets_unemployed_loss_next[type_idx, :, :],
                interpolation_method,
            )

            continuation_value_unemployed_loss[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_unemployed_loss_next[type_idx, :, :],
                hc_grid_reduced_e_a,
                assets_unemployed_loss_next[type_idx, :, :],
                interpolation_method,
            )

        pv_consumption_utility_unemployed_loss_now = (
            consumption_utility(policy_consumption_unemployed_loss[:, :, :, period_idx])
            + discount_factor
            * job_finding_probability(effort_searching_loss_next)
            * continuation_value_employed_loss
            + discount_factor
            * (1 - job_finding_probability(effort_searching_loss_next))
            * continuation_value_unemployed_loss
        )

        # unemployed
        assets_unemployed_next = np.maximum(
            (1 + interest_rate) * assets_grid_n_e_a
            + simulate_ui_benefits(
                wage_level * wage_hc_factor_grid_n_e_a,
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            )
            - policy_consumption_unemployed[:, :, :, period_idx],
            borrowing_limit_n_e_a,
        )

        effort_searching_next = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        effort_searching_loss_next = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_employed = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_employed_loss = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_unemployed = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_unemployed_loss = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        for type_idx in range(n_types):
            effort_searching_next[type_idx, :, :] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                policy_effort_searching[type_idx, :, :, period_idx + 1],
                hc_grid_reduced_e_a,
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )
            effort_searching_loss_next[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                policy_effort_searching_loss[type_idx, :, :, period_idx + 1],
                hc_grid_reduced_e_a,
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )

            continuation_value_employed[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_employed_next[type_idx, :, :],
                hc_grid_reduced_e_a,
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )
            continuation_value_employed_loss[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_employed_next[type_idx, :, :],
                _hc_after_loss_1_agent(
                    hc_grid_reduced_e_a,
                    wage_loss_factor_vector[type_idx, :],
                    wage_loss_reference_vector[type_idx, :],
                    period_idx + 1,
                ),
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )
            continuation_value_unemployed[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_unemployed_next[type_idx, :, :],
                hc_grid_reduced_e_a,
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )
            continuation_value_unemployed_loss[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_unemployed_loss_next[type_idx, :, :],
                hc_grid_reduced_e_a,
                assets_unemployed_next[type_idx, :, :],
                interpolation_method,
            )

        pv_consumption_utility_unemployed_now = (
            consumption_utility(policy_consumption_unemployed[:, :, :, period_idx])
            + discount_factor
            * np.repeat(
                (1 - hc_loss_probability), hc_grid_reduced_size * assets_grid_size
            ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
            * (
                job_finding_probability(effort_searching_next)
                * continuation_value_employed
                + (1 - job_finding_probability(effort_searching_next))
                * continuation_value_unemployed
            )
            + discount_factor
            * np.repeat(
                hc_loss_probability, hc_grid_reduced_size * assets_grid_size
            ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
            * (
                job_finding_probability(effort_searching_loss_next)
                * continuation_value_employed_loss
                + (1 - job_finding_probability(effort_searching_loss_next))
                * continuation_value_unemployed_loss
            )
        )

        # employed
        assets_employed_next = np.maximum(
            (1 + interest_rate) * assets_grid_n_e_a
            + np.repeat(
                (1 - income_tax_rate_vector[:, period_idx]),
                (hc_grid_reduced_size * assets_grid_size),
            ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
            * wage_level
            * wage_hc_factor_grid_n_e_a
            - policy_consumption_unemployed_loss[:, :, :, period_idx],
            borrowing_limit_n_e_a,
        )

        effort_searching_next = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_employed = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        continuation_value_unemployed = np.full(
            (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
        )
        for type_idx in range(n_types):
            effort_searching_next[type_idx, :, :] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                policy_effort_searching[type_idx, :, :, period_idx + 1],
                np.minimum(hc_grid_reduced_e_a + 1, hc_max),
                assets_employed_next[type_idx, :, :],
                interpolation_method,
            )

            continuation_value_employed[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_employed_next[type_idx, :, :],
                np.minimum(hc_grid_reduced_e_a + 1, hc_max),
                assets_employed_next[type_idx, :, :],
                interpolation_method,
            )
            continuation_value_unemployed[
                type_idx, :, :
            ] = interpolate_2d_ordered_to_unordered(
                hc_grid_reduced_e_a,
                assets_grid_e_a,
                pv_consumption_utility_unemployed_next[type_idx, :, :],
                np.minimum(hc_grid_reduced_e_a + 1, hc_max),
                assets_employed_next[type_idx, :, :],
                interpolation_method,
            )

        pv_consumption_utility_employed_now = (
            consumption_utility(policy_consumption_employed[:, :, :, period_idx])
            + discount_factor
            * np.repeat(
                (1 - separation_rate_vector[:, period_idx]),
                hc_grid_reduced_size * assets_grid_size,
            ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
            * continuation_value_employed
            + discount_factor
            * np.repeat(
                separation_rate_vector[:, period_idx],
                hc_grid_reduced_size * assets_grid_size,
            ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
            * continuation_value_unemployed
        )

        # initiate next iteration
        pv_consumption_utility_employed_next = pv_consumption_utility_employed_now
        pv_consumption_utility_unemployed_next = pv_consumption_utility_unemployed_now
        pv_consumption_utility_unemployed_loss_next = (
            pv_consumption_utility_unemployed_loss_now
        )

        period_idx -= 1

    # iteration complete
    pv_consumption_utility_searching = (
        policy_effort_searching[:, :, :, 0] * pv_consumption_utility_employed_now
        + (1 - policy_effort_searching[:, :, :, 0])
        * pv_consumption_utility_unemployed_now
    )

    pv_consumption_utility_at_entry = np.full(n_types, np.nan)
    for type_idx in range(n_types):
        pv_consumption_utility_at_entry[type_idx] = interpolate.interp1d(
            assets_grid,
            pv_consumption_utility_searching[type_idx, 0, :],
            kind=interpolation_method,
        )(0.0)

    pv_consumption_utility_at_entry = pd.DataFrame(
        data=pv_consumption_utility_at_entry,
        index=["high", "medium", "low"],
        columns=["pv consumption utility"],
    ).T
    pv_consumption_utility_at_entry.loc[
        "pv consumption utility", "overall"
    ] = np.average(
        pv_consumption_utility_at_entry.loc[
            "pv consumption utility", ["high", "medium", "low"]
        ],
        weights=type_weights,
    )

    # # solve for present value of consumption utility
    # period_idx = n_periods_working - 1
    #
    # consumption_diff = -npf.pmt(
    #     interest_rate,
    #     n_periods_retired + (n_periods_working - (period_idx + 1)),
    #     assets_max - np.amax(assets_grid),
    # )
    #
    # # unemployed with hc loss
    # assets_next_period_unemployed_loss = (
    #     assets_grid_e_a * (1 + interest_rate)
    #     + ui_replacement_rate_vector[period_idx] * wage_level * wage_hc_factor_e_a
    #     - policy_consumption_unemployed_loss[:, :, period_idx]
    # )
    #
    # value_unemployed_diff = -npf.pv(
    #     interest_rate,
    #     n_periods_retired + (n_periods_working - (period_idx + 1)),
    #     consumption_utility(cq[:, -1] + consumption_diff)
    #     - consumption_utility(cq[:, -1]),
    # )
    #
    # continuation_value_unemployed_loss = interpolate_2d_ordered_to_unordered(
    #     hc_grid_reduced_e_a1,
    #     assets_grid_e_a1,
    #     np.append(
    #         QtU[hc_grid_reduced, :],
    #         QtU[hc_grid_reduced, -1] + value_unemployed_diff[hc_grid_reduced, :],
    #         axis=1,
    #     ),
    #     hc_after_depreciation(
    #         hc_grid_reduced_e_a,
    #         period_idx + 1,
    #         wage_loss_factor_vector,
    #         wage_hc_factor_vector,
    #     ),
    #     assets_next_period_unemployed_loss,
    # )
    # pv_consumption_utility_unemployed_loss_now = (
    #     consumption_utility(policy_consumption_unemployed_loss[:, :, period_idx])
    #     + discount_factor * continuation_value_unemployed_loss
    # )
    #
    # # unemployed
    # assets_next_period = (
    #     assets_grid_e_a * (1 + interest_rate)
    #     + ui_replacement_rate_vector[n_periods_working]
    #     * wage_level
    #     * wage_hc_factor_e_a
    #     - policy_consumption_unemployed[:, :, period_idx]
    # )
    #
    # continuation_value_unemployed = np.full(
    #     (hc_grid_reduced_size, assets_grid_size), np.nan
    # )
    # for nn in range(hc_grid_reduced_size):
    #     continuation_value_unemployed[nn, :] = interpolate.interp1d(
    #         assets_grid,
    #         QtU[hc_grid_reduced[nn], :],
    #         kind=interpolation_method,
    #         bounds_error=False,
    #         fill_value="extrapolate",
    #     )(assets_next_period[nn, :])
    #
    # continuation_value_unemployed_loss = interpolate_2d_ordered_to_unordered(
    #     hc_grid_reduced_e_a1,
    #     assets_grid_e_a1,
    #     np.append(
    #         QtU[hc_grid_reduced, :],
    #         QtU[hc_grid_reduced, -1] + value_unemployed_diff[hc_grid_reduced, :],
    #         axis=1,
    #     ),
    #     hc_after_depreciation(
    #         hc_grid_reduced_e_a,
    #         period_idx + 1,
    #         wage_loss_factor_vector,
    #         wage_hc_factor_vector,
    #     ),
    #     assets_next_period,
    # )
    # pv_consumption_utility_unemployed_now = (
    #     consumption_utility(policy_consumption_unemployed[:, :, period_idx])
    #     + discount_factor * (1 - hc_loss_probability) * continuation_value_unemployed
    #     + discount_factor * hc_loss_probability * continuation_value_unemployed_loss
    # )
    #
    # # employed
    # assets_next_period = (
    #     assets_grid_e_a * (1 + interest_rate)
    #     + wage_level
    #     * wage_hc_factor_e_a
    #     * (1 - income_tax_rate_vector[n_periods_working])
    #     - policy_consumption_employed[:, :, period_idx]
    # )
    #
    # value_employed_diff = -npf.pv(
    #     interest_rate,
    #     n_periods_retired + (n_periods_working - (period_idx + 1)),
    #     consumption_utility(cq[:, -1] + consumption_diff)
    #     - consumption_utility(cq[:, -1]),
    # )
    #
    # continuation_value_employed = np.full(
    #     (hc_grid_reduced_size, assets_grid_size), np.nan
    # )
    # for nn in range(hc_grid_reduced_size):
    #     nnn = min(hc_grid_reduced[nn] + 1, n_periods_working + 1)
    #     continuation_value_employed[nn, :] = interpolate.interp1d(
    #         np.append(assets_grid, assets_max),
    #         np.append(Qt[nnn, :], Qt[nnn, -1] + value_employed_diff[nnn, -1]),
    #         kind=interpolation_method,
    #     )(assets_next_period[nn, :])
    #
    # pv_consumption_utility_employed_now = (
    #     consumption_utility(policy_consumption_employed[:, :, period_idx])
    #     + discount_factor * continuation_value_employed
    # )
    #
    # # searching
    # pv_consumption_utility_searching_now = (
    #     job_finding_probability(policy_effort_searching[:, :, period_idx])
    #     * pv_consumption_utility_employed_now
    #     + (1 - job_finding_probability(policy_effort_searching[:, :, period_idx]))
    #     * pv_consumption_utility_unemployed_now
    # )
    #
    # # searching with human capital loss
    # pv_consumption_utility_employed_loss_now = np.full(
    #     (hc_grid_reduced_size, assets_grid_size), np.nan
    # )
    # for i in range(assets_grid_size):
    #     pv_consumption_utility_employed_loss_now[:, i] = interpolate.interp1d(
    #         wage_hc_factor_vector[hc_grid_reduced],
    #         pv_consumption_utility_employed_now[:, i],
    #         kind=interpolation_method,
    #     )(
    #         hc_after_depreciation(
    #             hc_grid_reduced,
    #             n_periods_working,
    #             wage_loss_factor_vector,
    #             wage_hc_factor_vector,
    #         )
    #     )
    #
    # pv_consumption_utility_searching_loss_now = (
    #     job_finding_probability(policy_effort_searching_loss[:, :, period_idx])
    #     * pv_consumption_utility_employed_loss_now
    #     + (1 - job_finding_probability(policy_effort_searching_loss[:, :, period_idx]))
    #     * pv_consumption_utility_unemployed_loss_now
    # )
    #
    # # initiate next iteration
    # pv_consumption_utility_unemployed_loss_next = (
    #     pv_consumption_utility_unemployed_loss_now
    # )
    # pv_consumption_utility_unemployed_next = pv_consumption_utility_unemployed_now
    # pv_consumption_utility_employed_next = pv_consumption_utility_employed_now
    # pv_consumption_utility_searching_loss_next = (
    #     pv_consumption_utility_searching_loss_now
    # )
    # pv_consumption_utility_searching_next = pv_consumption_utility_searching_now
    #
    # period_idx -= 1
    #
    # while period_idx > 0:
    #
    #     # unemployed with human capital loss
    #     assets_next_period = np.maximum(
    #         (
    #             assets_grid_e_a * (1 + interest_rate)
    #             + ui_replacement_rate_vector[period_idx]
    #             * wage_level
    #             * wage_hc_factor_e_a
    #             - policy_consumption_unemployed_loss[:, :, period_idx]
    #         ),
    #         borrowing_limit_e_a,
    #     )
    #
    #     continuation_value_unemployed_loss = np.full(
    #         (hc_grid_reduced_size, assets_grid_size), np.nan
    #     )
    #     for nn in range(hc_grid_reduced_size):
    #         continuation_value_unemployed_loss[nn, :] = interpolate.interp1d(
    #             assets_grid,
    #             pv_consumption_utility_searching_loss_next[nn, :],
    #             kind=interpolation_method,
    #             bounds_error=False,
    #             fill_value="extrapolate",
    #         )(assets_next_period[nn, :])
    #
    #     pv_consumption_utility_unemployed_loss_now = (
    #         consumption_utility(policy_consumption_unemployed_loss[:, :, period_idx])
    #         + discount_factor * continuation_value_unemployed_loss
    #     )
    #
    #     # unemployed
    #     assets_next_period = max(
    #         (
    #             assets_grid_e_a * (1 + interest_rate)
    #             + ui_replacement_rate_vector[period_idx]
    #             * wage_level
    #             * wage_hc_factor_e_a
    #             - policy_consumption_unemployed[:, :, period_idx]
    #         ),
    #         borrowing_limit_e_a,
    #     )
    #
    #     continuation_value_searching = np.full(
    #         (hc_grid_reduced_size, assets_grid_size), np.nan
    #     )
    #     continuation_value_searching_loss = np.full(
    #         (hc_grid_reduced_size, assets_grid_size), np.nan
    #     )
    #     for nn in range(hc_grid_reduced_size):
    #         continuation_value_searching[nn, :] = interpolate.interp1d(
    #             assets_grid,
    #             pv_consumption_utility_searching_next[nn, :],
    #             kind=interpolation_method,
    #             bounds_error=False,
    #             fill_value="extrapolate",
    #         )(assets_next_period[nn, :])
    #         continuation_value_searching_loss[nn, :] = interpolate.interp1d(
    #             assets_grid,
    #             pv_consumption_utility_searching_loss_next[nn, :],
    #             kind=interpolation_method,
    #             bounds_error=False,
    #             fill_value="extrapolate",
    #         )(assets_next_period[nn, :])
    #
    #     pv_consumption_utility_unemployed_now = (
    #         consumption_utility(policy_consumption_unemployed[:, :, period_idx])
    #         + discount_factor * (1 - hc_loss_probability) * continuation_value_searching
    #         + discount_factor * hc_loss_probability * continuation_value_searching_loss
    #     )
    #
    #     # employed
    #     assets_next_period = max(
    #         (
    #             assets_grid_e_a * (1 + interest_rate)
    #             + wage_level
    #             * wage_hc_factor_e_a
    #             * (1 - income_tax_rate_vector[period_idx])
    #             - policy_consumption_employed[:, :, period_idx]
    #         ),
    #         borrowing_limit_e_a,
    #     )
    #
    #     value_employed_diff = -npf.pv(
    #         interest_rate,
    #         n_periods_retired + n_periods_working - period_idx,
    #         consumption_utility(
    #             policy_consumption_employed[:, -1, period_idx] + consumption_diff
    #         )
    #         - consumption_utility(policy_consumption_employed[:, -1, period_idx]),
    #     )
    #     value_unemployed_diff = -npf.pv(
    #         interest_rate,
    #         n_periods_retired + n_periods_working - period_idx,
    #         consumption_utility(
    #             policy_consumption_unemployed[:, -1, period_idx] + consumption_diff
    #         )
    #         - consumption_utility(policy_consumption_unemployed[:, -1, period_idx]),
    #     )
    #
    #     continuation_value_employed_plus = interpolate_2d_ordered_to_unordered(
    #         hc_grid_reduced_e_a1,
    #         assets_grid_e_a1,
    #         np.append(
    #             pv_consumption_utility_employed_next,
    #             pv_consumption_utility_employed_next[:, -1] + value_employed_diff,
    #         ),
    #         min(hc_grid_reduced_e_a + 1, 180),
    #         assets_next_period,
    #     )
    #     continuation_value_searching_plus = interpolate_2d_ordered_to_unordered(
    #         hc_grid_reduced_e_a1,
    #         assets_grid_e_a1,
    #         np.append(
    #             pv_consumption_utility_searching_next[1:hc_grid_reduced_size, :],
    #             pv_consumption_utility_searching_next[1:hc_grid_reduced_size, -1]
    #             + value_unemployed_diff,
    #         ),
    #         min(hc_grid_reduced_e_a + 1, 180),
    #         assets_next_period,
    #     )
    #
    #     pv_consumption_utility_employed_now = (
    #         consumption_utility(policy_consumption_employed[:, :, period_idx])
    #         + discount_factor
    #         * (1 - separation_rate_vector[period_idx])
    #         * continuation_value_employed_plus
    #         + discount_factor
    #         * separation_rate_vector(period_idx)
    #         * continuation_value_searching_plus
    #     )
    #
    #     # searching
    #     continuation_value_employed_loss = np.full(
    #         (hc_grid_reduced_size, assets_grid_size), np.nan
    #     )
    #     for i in range(assets_grid_size):
    #         continuation_value_employed_loss[:, i] = interpolate.interp1d(
    #             wage_hc_factor_vector[hc_grid_reduced],
    #             pv_consumption_utility_employed_now[:, i],
    #             interpolation_method,
    #             bounds_error=False,
    #             fill_value="extrapolate",
    #         )(
    #             hc_after_depreciation(
    #                 hc_grid_reduced,
    #                 period_idx,
    #                 wage_loss_factor_vector,
    #                 wage_hc_factor_vector,
    #             )
    #         )
    #
    #     pv_consumption_utility_searching_loss_now = (
    #         job_finding_probability(policy_effort_searching_loss[:, :, period_idx])
    #         * continuation_value_employed_loss
    #         + (
    #             1
    #             - job_finding_probability(
    #                 policy_effort_searching_loss[:, :, period_idx]
    #             )
    #         )
    #         * pv_consumption_utility_unemployed_loss_now
    #     )
    #     pv_consumption_utility_searching_now = (
    #         job_finding_probability(policy_effort_searching[:, :, period_idx])
    #         * pv_consumption_utility_employed_now
    #         + (1 - job_finding_probability(policy_effort_searching[:, :, period_idx]))
    #         * pv_consumption_utility_unemployed_now
    #     )
    #
    #     # initiate next iteration
    #     pv_consumption_utility_unemployed_loss_next = (
    #         pv_consumption_utility_unemployed_loss_now
    #     )
    #     pv_consumption_utility_unemployed_next = pv_consumption_utility_unemployed_now
    #     pv_consumption_utility_employed_next = pv_consumption_utility_employed_now
    #     pv_consumption_utility_searching_loss_next = (
    #         pv_consumption_utility_searching_loss_now
    #     )
    #     pv_consumption_utility_searching_next = pv_consumption_utility_searching_now
    #
    #     period_idx -= 1
    #
    # pv_consumption_utility = interpolate.interp1d(
    #     assets_grid,
    #     pv_consumption_utility_searching_now[1, :],
    #     kind=interpolation_method,
    # )(0.0)

    return pv_consumption_utility_at_entry


def _get_consumption_equivalents(setup_base, results_files, controls):

    # load calibration of baseline economy
    calibration_base = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_base + ".json"))
    )

    n_types = calibration_base["n_types"]
    type_weights = np.array(calibration_base["type_weights"])

    # load results of all setups
    results = {}
    for setup in results_files.keys():
        results[setup] = json.load(
            open(
                ppj(
                    "OUT_RESULTS",
                    "analytics",
                    "analytics_" + results_files[setup] + "_" + method + ".json",
                )
            )
        )
    results_base = results["Baseline"]

    # extract some variables
    risk_aversion_coefficient = calibration_base["risk_aversion_coefficient"]

    if n_types == 1:
        split = ["welfare_overall"]
    elif n_types == 3:
        split = ["welfare_high", "welfare_medium", "welfare_low"]

    welfare_all = pd.DataFrame(
        index=results_files.keys(),
        data=[results[setup]["welfare"] for setup in results_files.keys()],
        columns=split,
    )

    if n_types == 1:
        welfare_all.loc[:, "delta_overall"] = (
            welfare_all.loc[:, "welfare_overall"]
            - welfare_all.loc["Baseline", "welfare_overall"]
        )
    elif n_types == 3:
        welfare_all.loc[:, "welfare_overall"] = np.average(
            welfare_all.loc[:, split], weights=type_weights, axis=1
        )
        for group in ["overall", "high", "medium", "low"]:
            welfare_all.loc[:, "delta_" + group] = (
                welfare_all.loc[:, "welfare_" + group]
                - welfare_all.loc["Baseline", "welfare_" + group]
            )

    # compute present value of consumption utility in baseline economy
    pv_consumption_utility = _get_pv_consumption_utility(
        calibration_base, results_base, controls
    )

    # compute consumption equivalents for all setups
    if n_types == 1:
        welfare_all.loc[:, "consumption equivalent"] = (
            (
                1.0
                + welfare_all.loc[:, "delta to baseline"]
                / pv_consumption_utility.loc["pv consumption utility", "overall"]
            )
            ** (1.0 / (1.0 - risk_aversion_coefficient))
            - 1.0
        ) * 100.0
        welfare_all.loc[:, "consumption equivalent relative to first best"] = (
            welfare_all.loc[:, "consumption equivalent"]
            / welfare_all.loc["First best", "consumption equivalent"]
            - 1
        ) * 100
    elif n_types == 3:
        for group in ["overall", "high", "medium", "low"]:
            welfare_all.loc[:, "ce_" + group] = (
                (
                    1.0
                    + welfare_all.loc[:, "delta_" + group]
                    / pv_consumption_utility.loc["pv consumption utility", group]
                )
                ** (1.0 / (1.0 - risk_aversion_coefficient))
                - 1.0
            ) * 100.0
            welfare_all.loc[:, "ce_relative_" + group] = (
                welfare_all.loc[:, "ce_" + group]
                / welfare_all.loc["First best", "ce_" + group]
            ) * 100.0

    return welfare_all


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    try:
        method = sys.argv[1]
        equilibrium_condition = sys.argv[2]
    except IndexError:
        method = "linear"
        equilibrium_condition = "combined"

    analysis_plan = {
        "baseline": "base_" + equilibrium_condition,
        "results": {
            "First best": "base_" + equilibrium_condition + "_first_best",
            "Age and type dependent": "opt_rate_both_"
            + equilibrium_condition
            + "_results",
            "Age and type dependent (fixed budget)": "opt_rate_both_"
            + "fixed_budget"
            + "_results",
            "Age dependent": "opt_rate_age_" + equilibrium_condition + "_results",
            "Constant rate, floor and cap": "opt_rate_floor_cap_"
            + equilibrium_condition
            + "_results",
            "Baseline": "base_" + equilibrium_condition + "_results",
        },
    }

    setup_baseline = analysis_plan["baseline"]
    results_files = analysis_plan["results"]

    # set controls
    controls = {
        "interpolation_method": method,
    }

    # compute consumption equivalents
    consumption_equivalents = _get_consumption_equivalents(
        setup_baseline, results_files, controls
    )

    # store results
    consumption_equivalents.to_csv(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_welfare_comparison_"
            + equilibrium_condition
            + "_"
            + method
            + ".csv",
        )
    )
