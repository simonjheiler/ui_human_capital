import copy
import json
import sys
import warnings

import numba as nb
import numpy as np
import numpy_financial as npf
from bld.project_paths import project_paths_join as ppj
from scipy import interpolate

from src.utilities.interpolation_utils import interpolate_1d
from src.utilities.interpolation_utils import interpolate_2d_ordered_to_unordered
from src.utilities.interpolation_utils import interpolate_2d_unordered_to_unordered_iter
from src.utilities.interpolation_utils import interpolate_n_h_a_ordered_to_unordered


#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def conditional_mean(array, condition, axis):

    if axis == 0:
        signature = "i...,i... -> ..."
    elif axis == 1:
        signature = "ij..., ij... -> i..."
    elif axis == 2:
        signature = "ijk..., ijk... -> ij..."
    else:
        signature = None
        print("axis parameter unknown; select on of [0, 1, 2]")

    return np.einsum(signature, array, condition) / condition.sum(axis)


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


# @nb.njit
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


def _apply_borrowing_limit_employed(
    consumption_on_grid,
    assets_next,
    wage_hc_factor_grid,
    income_tax_rate_vector,
    period_idx,
):
    # adjust consumption for binding borrowing constraint
    consumption_corrected = (
        assets_next >= borrowing_limit_e_a
    ) * consumption_on_grid + (assets_next < borrowing_limit_e_a) * (
        assets_grid_e_a * (1 + interest_rate)
        + (1 - income_tax_rate_vector[period_idx]) * wage_level * wage_hc_factor_grid
        - borrowing_limit_e_a
    )

    # adjust assets for binding borrowing constraint
    assets_next_corrected = np.maximum(assets_next, borrowing_limit_e_a)

    return consumption_corrected, assets_next_corrected


def _apply_borrowing_limit_unemployed(
    consumption_on_grid,
    assets_next,
    wage_hc_factor_grid,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # adjust consumption for binding borrowing constraint
    consumption_corrected = (
        assets_next >= borrowing_limit_e_a
    ) * consumption_on_grid + (assets_next < borrowing_limit_e_a) * (
        assets_grid_e_a * (1 + interest_rate)
        + _ui_benefits(
            wage_level * wage_hc_factor_grid,
            ui_replacement_rate_vector,
            ui_floor,
            ui_cap,
            period_idx,
        )
        - borrowing_limit_e_a
    )

    # adjust assets for binding borrowing constraint
    assets_next_corrected = np.maximum(assets_next, borrowing_limit_e_a)

    return consumption_corrected, assets_next_corrected


def _foc_employed(
    policy_effort_searching_next,
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    separation_rate_vector,
    period_idx,
):
    # interpolate next period consumption and search effort policies
    # to account for hc increase
    policy_consumption_employed_plus_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_consumption_employed_next,
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        assets_grid_e_a,
        method=interpolation_method,
    )
    policy_consumption_unemployed_plus_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_consumption_unemployed_next,
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        assets_grid_e_a,
        method=interpolation_method,
    )
    policy_effort_searching_plus_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_effort_searching_next,
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        assets_grid_e_a,
        method=interpolation_method,
    )

    # consumption via FOC [hc x assets]
    consumption_employed_off_grid = _consumption_utility_dx_inverted(
        separation_rate_vector[period_idx]
        * (
            job_finding_probability(policy_effort_searching_plus_next)
            * _consumption_utility_dx(policy_consumption_employed_plus_next)
            + (1 - job_finding_probability(policy_effort_searching_plus_next))
            * _consumption_utility_dx(policy_consumption_unemployed_plus_next)
        )
        + (1 - separation_rate_vector[period_idx])
        * _consumption_utility_dx(policy_consumption_employed_plus_next)
    )

    return consumption_employed_off_grid


def _foc_unemployed(
    policy_effort_searching_next,
    policy_effort_searching_loss_next,
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    policy_consumption_unemployed_loss_next,
    hc_loss_probability,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):

    # interpolate next period consumption policies to account fo hc loss
    policy_consumption_employed_loss_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_consumption_employed_next,
        _hc_after_loss_1_agent(
            hc_grid_reduced_e_a,
            wage_loss_factor_vector,
            wage_loss_reference_vector,
            period_idx + 1,
        ),
        assets_grid_e_a,
        method=interpolation_method,
    )

    # back out optimal consumption via FOC
    consumption_unemployed_off_grid = _consumption_utility_dx_inverted(
        (1 - hc_loss_probability)
        * (
            job_finding_probability(policy_effort_searching_next)
            * _consumption_utility_dx(policy_consumption_employed_next)
            + (1 - job_finding_probability(policy_effort_searching_next))
            * _consumption_utility_dx(policy_consumption_unemployed_next)
        )
        + hc_loss_probability
        * (
            job_finding_probability(policy_effort_searching_loss_next)
            * _consumption_utility_dx(policy_consumption_employed_loss_next)
            + (1 - job_finding_probability(policy_effort_searching_loss_next))
            * _consumption_utility_dx(policy_consumption_unemployed_loss_next)
        )
    )

    return consumption_unemployed_off_grid


def _foc_unemployed_loss(
    policy_effort_searching_loss_next,
    policy_consumption_employed_next,
    policy_consumption_unemployed_loss_next,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):

    # interpolate next period consumption policies to account fo hc loss
    policy_consumption_employed_loss_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_consumption_employed_next,
        _hc_after_loss_1_agent(
            hc_grid_reduced_e_a,
            wage_loss_factor_vector,
            wage_loss_reference_vector,
            period_idx + 1,
        ),
        assets_grid_e_a,
        method=interpolation_method,
    )

    # back out optimal consumption via FOC
    consumption_unemployed_loss_off_grid = _consumption_utility_dx_inverted(
        job_finding_probability(policy_effort_searching_loss_next)
        * _consumption_utility_dx(policy_consumption_employed_loss_next)
        + (1 - job_finding_probability(policy_effort_searching_loss_next))
        * _consumption_utility_dx(policy_consumption_unemployed_loss_next)
    )

    return consumption_unemployed_loss_off_grid


def _get_cost_employed(
    cost_employed_next,
    cost_unemployed_next,
    policy_consumption_employed_now,
    policy_effort_searching_next,
    policy_assets_employed_now,
    separation_rate_vector,
    wage_hc_factor_vector,
    consumption_tax_rate,
    income_tax_rate_vector,
    period_idx,
):

    # interpolate next period cost functions and search effort policy
    # to account for increase in hc and choice of assets next period
    cost_employed_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a1,
        assets_grid_e_a1,
        np.append(cost_employed_next, cost_employed_next[:, -1, np.newaxis], axis=1),
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        policy_assets_employed_now,
        method=interpolation_method,
    )
    cost_unemployed_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a1,
        assets_grid_e_a1,
        np.append(
            cost_unemployed_next, cost_unemployed_next[:, -1, np.newaxis], axis=1
        ),
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        policy_assets_employed_now,
        method=interpolation_method,
    )
    effort_searching_plus_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a1,
        assets_grid_e_a1,
        np.append(
            policy_effort_searching_next,
            policy_effort_searching_next[:, -1, np.newaxis],
            axis=1,
        ),
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        policy_assets_employed_now,
        method=interpolation_method,
    )

    # calculate current period cost
    cost_employed = (
        -consumption_tax_rate * policy_consumption_employed_now
        - np.repeat(
            (
                income_tax_rate_vector[period_idx]
                * wage_level
                * wage_hc_factor_vector[hc_grid_reduced]
            ),
            assets_grid_size,
        ).reshape(hc_grid_reduced_size, assets_grid_size)
        + discount_factor
        * (1 - separation_rate_vector[period_idx])
        * cost_employed_next_interpolated
        + discount_factor
        * separation_rate_vector[period_idx]
        * effort_searching_plus_next
        * cost_employed_next_interpolated
        + discount_factor
        * separation_rate_vector[period_idx]
        * (1 - effort_searching_plus_next)
        * cost_unemployed_next_interpolated
    )

    return cost_employed


def _get_cost_unemployed(
    cost_employed_next,
    cost_unemployed_next,
    cost_unemployed_loss_next,
    policy_consumption_unemployed_now,
    policy_effort_searching_next,
    policy_effort_searching_loss_next,
    policy_assets_unemployed_now,
    hc_loss_probability,
    wage_hc_factor_vector,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    consumption_tax_rate,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # interpolate next period cost functions and search effort policy
    # to account for hc loss and choice of assets next period
    cost_employed_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_employed_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_now,
        method=interpolation_method,
    )
    cost_unemployed_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_unemployed_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_now,
        method=interpolation_method,
    )
    cost_unemployed_loss_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_unemployed_loss_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_now,
        method=interpolation_method,
    )
    cost_employed_loss_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_employed_next,
        _hc_after_loss_1_agent(
            hc_grid_reduced_e_a,
            wage_loss_factor_vector,
            wage_loss_reference_vector,
            period_idx + 1,
        ),
        policy_assets_unemployed_now,
        method=interpolation_method,
    )
    effort_searching_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_effort_searching_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_now,
        method=interpolation_method,
    )
    effort_searching_loss_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_effort_searching_loss_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_now,
        method=interpolation_method,
    )

    # calculate current period cost
    cost_unemployed = (
        -consumption_tax_rate * policy_consumption_unemployed_now
        + np.repeat(
            _ui_benefits(
                wage_level * wage_hc_factor_vector[hc_grid_reduced],
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            ),
            assets_grid_size,
        ).reshape(hc_grid_reduced_size, assets_grid_size)
        + discount_factor
        * (1 - hc_loss_probability)
        * (
            job_finding_probability(effort_searching_next)
            * cost_employed_next_interpolated
            + (1 - job_finding_probability(effort_searching_next))
            * cost_unemployed_next_interpolated
        )
        + discount_factor
        * hc_loss_probability
        * (
            job_finding_probability(effort_searching_loss_next)
            * cost_employed_loss_next_interpolated
            + (1 - job_finding_probability(effort_searching_loss_next))
            * cost_unemployed_loss_next_interpolated
        )
    )

    return cost_unemployed


def _get_cost_unemployed_loss(
    cost_employed_next,
    cost_unemployed_loss_next,
    policy_consumption_unemployed_loss_now,
    policy_effort_searching_loss_next,
    policy_assets_unemployed_loss_now,
    wage_hc_factor_vector,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    consumption_tax_rate,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # interpolate next period cost functions and search effort policy
    # to account for hc loss and choice of assets next period
    cost_employed_loss_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_employed_next,
        _hc_after_loss_1_agent(
            hc_grid_reduced_e_a,
            wage_loss_factor_vector,
            wage_loss_reference_vector,
            period_idx + 1,
        ),
        policy_assets_unemployed_loss_now,
        method=interpolation_method,
    )
    cost_unemployed_loss_next_interpolated = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        cost_unemployed_loss_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_loss_now,
        method=interpolation_method,
    )
    effort_searching_loss_next = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        policy_effort_searching_loss_next,
        hc_grid_reduced_e_a,
        policy_assets_unemployed_loss_now,
        method=interpolation_method,
    )

    # calculate current period cost
    cost_unemployed_loss = (
        -consumption_tax_rate * policy_consumption_unemployed_loss_now
        + np.repeat(
            _ui_benefits(
                wage_level * wage_hc_factor_vector[hc_grid_reduced],
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            ),
            assets_grid_size,
        ).reshape(hc_grid_reduced_size, assets_grid_size)
        + discount_factor
        * (
            effort_searching_loss_next * cost_employed_loss_next_interpolated
            + (1 - effort_searching_loss_next) * cost_unemployed_loss_next_interpolated
        )
    )

    return cost_unemployed_loss


@nb.njit
def _get_duration_weeks(
    job_finding_probability_quarter,
    duration_quarter,
):

    # transform quarterly job finding probability to weekly job finding probability
    job_finding_probability_week = 1 - (1 - job_finding_probability_quarter) ** (1 / 13)

    # calculate expected additional unemployment duration in weeks
    additional_duration_weeks = (
        (1 - job_finding_probability_week)
        - (1 + job_finding_probability_week * 12)
        * (1 - job_finding_probability_week) ** 13
    ) / (job_finding_probability_week * (1 - (1 - job_finding_probability_week) ** 13))

    # correct for small probabilities (expression approaches 6 in the limit,
    # but gets numerically unstable for small probabilities)
    additional_duration_weeks = (
        job_finding_probability_quarter > 0.001
    ) * additional_duration_weeks + (
        job_finding_probability_quarter <= 0.001
    ) * np.full_like(
        additional_duration_weeks, 6.0
    )

    # transform quarterly duration to weekly duration
    duration_weeks = duration_quarter * 13

    # add expected additional weeks
    duration_weeks += additional_duration_weeks

    return duration_weeks


def _get_value_employed(
    policy_consumption_employed_now,
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    value_searching_next,
    value_employed_next,
    assets_next,
    separation_rate_vector,
    period_idx,
):
    # expand interpolation grid
    consumption_diff = -npf.pmt(
        interest_rate,
        n_periods_retired + (n_periods_working - (period_idx + 1)),
        assets_max - np.amax(assets_grid),
    )
    value_employed_diff = -npf.pv(
        interest_rate,
        n_periods_retired + (n_periods_working - (period_idx + 1)),
        consumption_utility(policy_consumption_employed_next[:, -1] + consumption_diff)
        - consumption_utility(policy_consumption_employed_next[:, -1]),
    )
    value_unemployed_diff = -npf.pv(
        interest_rate,
        n_periods_retired + (n_periods_working - (period_idx + 1)),
        consumption_utility(
            policy_consumption_unemployed_next[:, -1] + consumption_diff
        )
        - consumption_utility(policy_consumption_unemployed_next[:, -1]),
    )

    # interpolate continuation value on grid to reflect increase in
    # hc and choice for assets next period
    continuation_value_employed = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a1,
        assets_grid_e_a1,
        np.append(
            value_employed_next,
            (value_employed_next[:, -1] + value_employed_diff)[..., np.newaxis],
            axis=1,
        ),
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        assets_next,
        method=interpolation_method,
    )
    continuation_value_searching = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a1,
        assets_grid_e_a1,
        np.append(
            value_searching_next,
            (value_searching_next[:, -1] + value_unemployed_diff)[..., np.newaxis],
            axis=1,
        ),
        np.minimum(hc_grid_reduced_e_a + 1, hc_max),
        assets_next,
        method=interpolation_method,
    )

    # calculate value function from consumption level and continuation value
    value = (
        consumption_utility(policy_consumption_employed_now)
        + discount_factor
        * (1 - separation_rate_vector[period_idx])
        * continuation_value_employed
        + discount_factor
        * separation_rate_vector[period_idx]
        * continuation_value_searching
    )

    return value


def _get_value_unemployed(
    policy_consumption_unemployed_now,
    value_searching_next,
    value_searching_loss_next,
    assets_next,
    wage_loss_probability,
):
    # interpolate continuation value on grid to reflect choice for
    # assets next period
    continuation_value_searching = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        value_searching_next,
        hc_grid_reduced_e_a,
        assets_next,
        method=interpolation_method,
    )
    continuation_value_searching_loss = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        value_searching_loss_next,
        hc_grid_reduced_e_a,
        assets_next,
        method=interpolation_method,
    )

    # calculate value function from consumption level and continuation value
    value = (
        consumption_utility(policy_consumption_unemployed_now)
        + discount_factor * (1 - wage_loss_probability) * continuation_value_searching
        + discount_factor * wage_loss_probability * continuation_value_searching_loss
    )

    return value


def _get_value_unemployed_loss(
    policy_consumption,
    value_searching_next,
    assets_next,
):
    # interpolate continuation value on grid
    continuation_value_searching_loss = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        value_searching_next,
        hc_grid_reduced_e_a,
        assets_next,
        method=interpolation_method,
    )

    # calculate value function from consumption level and continuation value
    value = (
        consumption_utility(policy_consumption)
        + discount_factor * continuation_value_searching_loss
    )

    return value


def _interpolate_consumption_on_grid(
    consumption_off_grid,
    assets_off_grid,
):
    # interpolate consumption on grid
    consumption_on_grid = interpolate_2d_unordered_to_unordered_iter(
        hc_grid_reduced_e_a,
        assets_off_grid.astype(float),
        consumption_off_grid.astype(float),
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        method=interpolation_method,
    )

    # # interpolate consumption on grid
    # consumption_on_grid = interpolate_2d_unordered_to_unordered(
    #     hc_grid_reduced_e_a,
    #     assets_off_grid,
    #     consumption_off_grid,
    #     hc_grid_reduced_e_a,
    #     assets_grid_e_a,
    #     method=method_2d
    # )
    # # fill values outside convex hull with nearest neighbor
    # if np.any(np.isnan(consumption_on_grid)):
    #     consumption_fill = interpolate_2d_unordered_to_unordered(
    #         hc_grid_reduced_e_a,
    #         assets_off_grid,
    #         consumption_off_grid,
    #         hc_grid_reduced_e_a,
    #         assets_grid_e_a,
    #         method="nearest",
    #     )
    #     consumption_on_grid[np.isnan(consumption_on_grid)] = consumption_fill[
    #         np.isnan(consumption_on_grid)
    #     ]

    return consumption_on_grid


def _invert_bc_employed(
    consumption_off_grid,
    wage_hc_factor_grid,
    income_tax_rate_vector,
    period_idx,
):

    assets_off_grid = discount_factor * (
        assets_grid_e_a
        - (1 - income_tax_rate_vector[period_idx]) * wage_level * wage_hc_factor_grid
        + consumption_off_grid
    )

    return assets_off_grid


def _invert_bc_unemployed(
    consumption_off_grid,
    wage_hc_factor_grid,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    benefits = _ui_benefits(
        wage_level * wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    assets_off_grid = discount_factor * (
        assets_grid_e_a - benefits + consumption_off_grid
    )

    return assets_off_grid


def _solve_bc_employed(
    consumption_now,
    wage_hc_factor_grid,
    income_tax_rate_vector,
    period_idx,
):

    assets_next = (
        assets_grid_e_a * (1 + interest_rate)
        + (1 - income_tax_rate_vector[period_idx]) * wage_level * wage_hc_factor_grid
        - consumption_now
    )

    return assets_next


def _solve_bc_unemployed(
    consumption_now,
    wage_hc_factor_grid,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    benefits = _ui_benefits(
        wage_level * wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    assets_next = assets_grid_e_a * (1 + interest_rate) + benefits - consumption_now

    return assets_next


def _solve_one_period(
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    policy_consumption_unemployed_loss_next,
    policy_effort_searching_next,
    policy_effort_searching_loss_next,
    value_employed_next,
    value_searching_next,
    value_searching_loss_next,
    cost_employed_next,
    cost_unemployed_next,
    cost_unemployed_loss_next,
    hc_loss_probability,
    separation_rate_vector,
    wage_hc_factor_grid,
    wage_hc_factor_vector,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    consumption_tax_rate,
    income_tax_rate_vector,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):
    # SOLVE FOR: unemployed after hc loss
    (
        policy_consumption_unemployed_loss_now,
        policy_assets_unemployed_loss_now,
        value_unemployed_loss_now,
    ) = _solve_unemployed_loss(
        policy_consumption_employed_next,
        policy_consumption_unemployed_loss_next,
        policy_effort_searching_loss_next,
        value_searching_loss_next,
        wage_hc_factor_grid,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # SOLVE FOR: unemployed
    (
        policy_consumption_unemployed_now,
        policy_assets_unemployed_now,
        value_unemployed_now,
    ) = _solve_unemployed(
        policy_consumption_employed_next,
        policy_consumption_unemployed_next,
        policy_consumption_unemployed_loss_next,
        policy_effort_searching_next,
        policy_effort_searching_loss_next,
        value_searching_next,
        value_searching_loss_next,
        hc_loss_probability,
        wage_hc_factor_grid,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # SOLVE FOR: employed
    (
        policy_consumption_employed_now,
        policy_assets_employed_now,
        value_employed_now,
    ) = _solve_employed(
        policy_consumption_employed_next,
        policy_consumption_unemployed_next,
        policy_effort_searching_next,
        value_employed_next,
        value_searching_next,
        separation_rate_vector,
        wage_hc_factor_grid,
        income_tax_rate_vector,
        period_idx,
    )

    # SOLVE FOR: searching and searching with hc loss
    (
        policy_effort_searching_now,
        policy_effort_searching_loss_now,
        value_searching_now,
        value_searching_loss_now,
    ) = _solve_searching(
        value_employed_now,
        value_unemployed_now,
        value_unemployed_loss_now,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        period_idx,
    )

    # update cost functions
    cost_unemployed_loss_now = _get_cost_unemployed_loss(
        cost_employed_next,
        cost_unemployed_loss_next,
        policy_consumption_unemployed_loss_now,
        policy_effort_searching_loss_next,
        policy_assets_unemployed_loss_now,
        wage_hc_factor_vector,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        consumption_tax_rate,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )
    cost_unemployed_now = _get_cost_unemployed(
        cost_employed_next,
        cost_unemployed_next,
        cost_unemployed_loss_next,
        policy_consumption_unemployed_now,
        policy_effort_searching_next,
        policy_effort_searching_loss_next,
        policy_assets_unemployed_now,
        hc_loss_probability,
        wage_hc_factor_vector,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        consumption_tax_rate,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )
    cost_employed_now = _get_cost_employed(
        cost_employed_next,
        cost_unemployed_next,
        policy_consumption_employed_now,
        policy_effort_searching_next,
        policy_assets_employed_now,
        separation_rate_vector,
        wage_hc_factor_vector,
        consumption_tax_rate,
        income_tax_rate_vector,
        period_idx,
    )

    return (
        policy_consumption_employed_now,
        policy_consumption_unemployed_now,
        policy_consumption_unemployed_loss_now,
        policy_effort_searching_now,
        policy_effort_searching_loss_now,
        value_employed_now,
        value_unemployed_now,
        value_unemployed_loss_now,
        value_searching_now,
        value_searching_loss_now,
        cost_employed_now,
        cost_unemployed_now,
        cost_unemployed_loss_now,
    )


def _solve_employed(
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    policy_effort_searching_next,
    value_employed_next,
    value_searching_next,
    separation_rate_vector,
    wage_hc_factor_grid,
    income_tax_rate_vector,
    period_idx,
):

    # consumption from FOC [hc x assets]
    consumption_employed_off_grid = _foc_employed(
        policy_effort_searching_next,
        policy_consumption_employed_next,
        policy_consumption_unemployed_next,
        separation_rate_vector,
        period_idx,
    )

    # back out implicit current period asset levels (off grid)
    # for next period asset levels (on grid) [hc x assets]
    assets_off_grid = _invert_bc_employed(
        consumption_employed_off_grid,
        wage_hc_factor_grid,
        income_tax_rate_vector,
        period_idx,
    )

    # interpolate consumption levels for (on grid) asset levels [hc x assets]
    consumption_employed_on_grid = _interpolate_consumption_on_grid(
        consumption_employed_off_grid, assets_off_grid
    )

    # back out implicit asset level next period from budget constraint
    assets_employed_next = _solve_bc_employed(
        consumption_employed_on_grid,
        wage_hc_factor_grid,
        income_tax_rate_vector,
        period_idx,
    )

    # correct consumption for binding borrowing limit
    (
        policy_consumption_employed_now,
        assets_employed_next,
    ) = _apply_borrowing_limit_employed(
        consumption_employed_on_grid,
        assets_employed_next,
        wage_hc_factor_grid,
        income_tax_rate_vector,
        period_idx,
    )

    # calculate value function
    value_employed_now = _get_value_employed(
        policy_consumption_employed_now,
        policy_consumption_employed_next,
        policy_consumption_unemployed_next,
        value_searching_next,
        value_employed_next,
        assets_employed_next,
        separation_rate_vector,
        period_idx,
    )

    return policy_consumption_employed_now, assets_employed_next, value_employed_now


def _solve_unemployed(
    policy_consumption_employed_next,
    policy_consumption_unemployed_next,
    policy_consumption_unemployed_loss_next,
    policy_effort_searching_next,
    policy_effort_searching_loss_next,
    value_searching_next,
    value_searching_loss_next,
    hc_loss_probability,
    wage_hc_factor_grid,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # consumption via FOC [hc x assets]
    consumption_unemployed_off_grid = _foc_unemployed(
        policy_effort_searching_next,
        policy_effort_searching_loss_next,
        policy_consumption_employed_next,
        policy_consumption_unemployed_next,
        policy_consumption_unemployed_loss_next,
        hc_loss_probability,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        period_idx,
    )

    # back out implicit current period asset levels (off grid)
    # for next period asset levels (on grid)
    assets_off_grid = _invert_bc_unemployed(
        consumption_unemployed_off_grid,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # interpolate consumption levels for (on grid) asset levels
    consumption_unemployed_on_grid = _interpolate_consumption_on_grid(
        consumption_unemployed_off_grid, assets_off_grid
    )

    # back out implicit asset level next period from budget constraint
    assets_unemployed_next = _solve_bc_unemployed(
        consumption_unemployed_on_grid,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # correct consumption and asset policy for binding borrowing limit
    (
        policy_consumption_unemployed_now,
        policy_assets_unemployed_now,
    ) = _apply_borrowing_limit_unemployed(
        consumption_unemployed_on_grid,
        assets_unemployed_next,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # calculate value function
    value_unemployed_now = _get_value_unemployed(
        policy_consumption_unemployed_now,
        value_searching_next,
        value_searching_loss_next,
        policy_assets_unemployed_now,
        hc_loss_probability,
    )

    return (
        policy_consumption_unemployed_now,
        policy_assets_unemployed_now,
        value_unemployed_now,
    )


def _solve_unemployed_loss(
    policy_consumption_employed_next,
    policy_consumption_unemployed_loss_next,
    policy_effort_searching_loss_next,
    value_searching_loss_next,
    wage_hc_factor_grid,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # consumption via FOC [hc level x asset level]
    consumption_unemployed_loss_off_grid = _foc_unemployed_loss(
        policy_effort_searching_loss_next,
        policy_consumption_employed_next,
        policy_consumption_unemployed_loss_next,
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        period_idx,
    )

    # back out implicit current period asset levels (off grid)
    # for next period asset levels (on grid)
    assets_off_grid = _invert_bc_unemployed(
        consumption_unemployed_loss_off_grid,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # interpolate consumption levels for (on grid) asset levels
    consumption_unemployed_loss_on_grid = _interpolate_consumption_on_grid(
        consumption_unemployed_loss_off_grid,
        assets_off_grid,
    )

    # back out implicit asset level next period from budget constraint
    assets_unemployed_loss_next = _solve_bc_unemployed(
        consumption_unemployed_loss_on_grid,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # correct consumption and asset policy for binding borrowing limit
    (
        policy_consumption_unemployed_loss_now,
        policy_assets_unemployed_loss_now,
    ) = _apply_borrowing_limit_unemployed(
        consumption_unemployed_loss_on_grid,
        assets_unemployed_loss_next,
        wage_hc_factor_grid,
        ui_replacement_rate_vector,
        ui_floor,
        ui_cap,
        period_idx,
    )

    # calculate value function
    value_unemployed_loss_now = _get_value_unemployed_loss(
        policy_consumption_unemployed_loss_now,
        value_searching_loss_next,
        policy_assets_unemployed_loss_now,
    )

    return (
        policy_consumption_unemployed_loss_now,
        policy_assets_unemployed_loss_now,
        value_unemployed_loss_now,
    )


def _solve_searching(
    value_employed_now,
    value_unemployed_now,
    value_unemployed_loss_now,
    wage_loss_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):

    # interpolate value of being employed at depreciated hc levels
    value_employed_loss_now = interpolate_2d_ordered_to_unordered(
        hc_grid_reduced_e_a,
        assets_grid_e_a,
        value_employed_now,
        _hc_after_loss_1_agent(
            hc_grid_reduced_e_a,
            wage_loss_factor_vector,
            wage_loss_reference_vector,
            period_idx,
        ),
        assets_grid_e_a,
        method=interpolation_method,
    )

    # solve using grid search
    (policy_effort_searching_now, value_searching_now) = _solve_searching_iter(
        value_employed_now, value_unemployed_now
    )
    (
        policy_effort_searching_loss_now,
        value_searching_loss_now,
    ) = _solve_searching_iter(value_employed_loss_now, value_unemployed_loss_now)

    # solve using FOCs
    # (
    #     policy_effort_searching_now, value_searching_now
    # ) = _solve_searching_foc(value_employed_now, value_unemployed_now)
    # (
    #     policy_effort_searching_loss_now, value_searching_loss_now
    # ) = _solve_searching_foc(value_employed_loss_now, value_unemployed_loss_now)

    return (
        policy_effort_searching_now,
        policy_effort_searching_loss_now,
        value_searching_now,
        value_searching_loss_now,
    )


def _solve_searching_base(value_employed, value_unemployed):
    # initiate objects
    policy = np.full((hc_grid_reduced_size, assets_grid_size), np.full)
    value = np.full((hc_grid_reduced_size, assets_grid_size), np.full)

    # solve for optimal search effort using grid search method
    for asset_level in range(assets_grid_size):
        search_returns = (
            np.tile(
                leisure_utility_interpolated(search_effort_grid),
                hc_grid_reduced_size,
            ).reshape(hc_grid_reduced_size, search_effort_grid_size)
            + job_finding_probability_grid
            * np.repeat(
                value_employed[:, asset_level], search_effort_grid_size
            ).reshape(hc_grid_reduced_size, search_effort_grid_size)
            + (1 - job_finding_probability_grid)
            * np.repeat(
                value_unemployed[:, asset_level], search_effort_grid_size
            ).reshape(hc_grid_reduced_size, search_effort_grid_size)
        )

        search_effort_idx = search_returns.argmax(axis=1)
        value[:, asset_level] = np.array(
            [search_returns[row, col] for row, col in enumerate(search_effort_idx)]
        )
        policy[:, asset_level] = search_effort_grid[search_effort_idx].T

    return policy, value


def on_grid(x):
    return search_effort_grid[np.abs(x - search_effort_grid).argmin()]


def on_grid_vectorized(x):
    return np.vectorize(on_grid)(x)


@nb.njit
def on_grid_iter(array_in):

    dims = array_in.shape

    array_out = np.full(array_in.shape, np.nan)

    for x_idx in range(dims[0]):
        for y_idx in range(dims[1]):
            array_out[x_idx, y_idx] = search_effort_grid[
                np.abs(array_in[x_idx, y_idx] - search_effort_grid).argmin()
            ]

    return array_out


def _solve_searching_foc(value_employed, value_unemployed):

    # optimal effort from FOC / constraints
    effort_off_grid = leisure_utility_dx_inverted(value_unemployed - value_employed)
    effort_off_grid[
        (value_unemployed - value_employed) > leisure_utility_dx_min
    ] = search_effort_grid[0]
    effort_off_grid[
        (value_unemployed - value_employed) < leisure_utility_dx_max
    ] = search_effort_grid[-1]

    # get nearest values on grid
    policy = on_grid_iter(effort_off_grid)
    value = (
        leisure_utility_interpolated(policy)
        + job_finding_probability(policy) * value_employed
        + (1 - job_finding_probability(policy)) * value_unemployed
    )

    return policy.astype(float), value.astype(float)


def _solve_searching_vectorized(
    value_employed,
    value_unemployed,
):
    # solve for optimal search effort using grid search method
    search_returns = (
        np.repeat(
            leisure_utility_interpolated(search_effort_grid),
            hc_grid_reduced_size * assets_grid_size,
        )
        .reshape((search_effort_grid_size, assets_grid_size, hc_grid_reduced_size))
        .T
        + np.repeat(
            job_finding_probability_grid,
            hc_grid_reduced_size * assets_grid_size,
        )
        .reshape((search_effort_grid_size, assets_grid_size, hc_grid_reduced_size))
        .T
        * np.tile(value_employed, search_effort_grid_size)
        .T.reshape(search_effort_grid_size, assets_grid_size, hc_grid_reduced_size)
        .T
        + np.repeat(
            (1 - job_finding_probability_grid),
            hc_grid_reduced_size * assets_grid_size,
        )
        .reshape((search_effort_grid_size, assets_grid_size, hc_grid_reduced_size))
        .T
        * np.tile(value_unemployed, search_effort_grid_size)
        .T.reshape((search_effort_grid_size, assets_grid_size, hc_grid_reduced_size))
        .T
    )

    search_effort_idx = np.argmax(search_returns, axis=2)
    value = np.amax(search_returns, axis=2)
    policy = search_effort_grid[search_effort_idx]

    return policy, value


@nb.njit
def _solve_searching_iter(
    value_employed_now,
    value_unemployed_now,
):
    # initiate objects
    policy = np.full((hc_grid_reduced_size, assets_grid_size), np.nan)
    value = np.full((hc_grid_reduced_size, assets_grid_size), np.nan)

    # solve for optimal search effort using grid search method
    for hc_level in range(hc_grid_reduced_size):
        for asset_level in range(assets_grid_size):

            search_returns = (
                leisure_utility_on_search_grid
                + job_finding_probability_grid
                * np.full(
                    search_effort_grid_size,
                    value_employed_now[hc_level, asset_level],
                )
                + (1 - job_finding_probability_grid)
                * np.full(
                    search_effort_grid_size,
                    value_unemployed_now[hc_level, asset_level],
                )
            )
            search_effort_idx = search_returns.argmax()
            value[hc_level, asset_level] = search_returns[search_effort_idx]
            policy[hc_level, asset_level] = search_effort_grid[search_effort_idx]

    return policy, value


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


def leisure_utility_dx_interpolated(x):
    return interpolate.interp1d(
        leisure_grid, leisure_utility_dx, kind=interpolation_method
    )(x)


def leisure_utility_dxdx_interpolated(x):
    return interpolate.interp1d(
        leisure_grid, leisure_utility_dxdx, kind=interpolation_method
    )(x)


def leisure_utility_interpolated(x):
    return interpolate.interp1d(
        leisure_grid, leisure_utility, kind=interpolation_method
    )(x)


def leisure_utility_dx_inverted(x):
    return interpolate.interp1d(
        leisure_utility_dx,
        leisure_grid,
        kind=interpolation_method,
        bounds_error=False,
        fill_value=np.nan,
    )(x)


@nb.njit
def _consumption_utility_dx(x):
    return x ** (-risk_aversion_coefficient)


@nb.njit
def _consumption_utility_dx_inverted(x):

    return x ** (-1 / risk_aversion_coefficient)


def wage_hc_factor_interpolated_1_agent(x, wage_hc_factor_vector):
    return interpolate.interp1d(
        hc_grid,
        wage_hc_factor_vector,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(x)


def _simulate_transition_consumption_searching(
    employed_simulated_now,
    unemployed_simulated_now,
    unemployed_loss_simulated_now,
    nonemployed_simulated_now,
    hc_simulated,
    duration_unemployed_simulated,
    duration_employed_simulated,
    period_idx,
    separation_rate_vector,
    wage_loss_probability,
):

    # simulate transition events
    job_loss_event_simulated = np.array(
        [
            separation_rate_vector[i, period_idx] > np.random.rand(n_simulations)
            for i in range(n_types)
        ]
    )
    hc_loss_event_simulated = np.array(
        [
            wage_loss_probability[i] >= np.random.rand(n_simulations)
            for i in range(n_types)
        ]
    )

    # simulate transitions in employment status
    employed_simulated_next = employed_simulated_now * (
        1 - job_loss_event_simulated
    ).astype(bool)
    searching_simulated_next = (
        unemployed_simulated_now * (1 - hc_loss_event_simulated)
        + employed_simulated_now * job_loss_event_simulated
    ).astype(bool)
    searching_loss_simulated_next = (
        unemployed_loss_simulated_now
        + unemployed_simulated_now * hc_loss_event_simulated
    ).astype(bool)

    # simulate experience transition
    hc_simulated[
        employed_simulated_now
    ] += 1  # increase experience of employed workers by 1

    # update duration tracker
    duration_unemployed_simulated[
        nonemployed_simulated_now
    ] += 1  # increase duration of unemployed (with / without hc loss) by 1
    duration_unemployed_simulated[
        employed_simulated_now
    ] = 0  # set duration of now employed to 0
    duration_employed_simulated[employed_simulated_now] += 1
    duration_employed_simulated[nonemployed_simulated_now] = 0

    return (
        employed_simulated_next,
        searching_simulated_next,
        searching_loss_simulated_next,
        duration_unemployed_simulated,
        duration_employed_simulated,
        hc_simulated,
    )


def _simulate_consumption(
    policy_consumption_employed,
    policy_consumption_unemployed,
    policy_consumption_unemployed_loss,
    employed_simulated,
    unemployed_simulated,
    unemployed_loss_simulated,
    hc_simulated,
    assets_simulated,
    period_idx,
):

    # calculate consumption differential to increase asset grid
    consumption_diff = -npf.pmt(
        interest_rate,
        n_periods_retired + (n_periods_working - period_idx + 1),
        assets_max - max(assets_grid),
    )

    # initiate intermediate objects
    consumption_employed_simulated = np.full((n_types, n_simulations), np.nan)
    consumption_unemployed_simulated = np.full((n_types, n_simulations), np.nan)
    consumption_unemployed_loss_simulated = np.full((n_types, n_simulations), np.nan)

    # interpolate consumption policies
    for type_idx in range(n_types):
        consumption_unemployed_simulated[
            type_idx, :
        ] = interpolate_2d_ordered_to_unordered(
            hc_grid_reduced_e_a1,
            assets_grid_e_a1,
            np.append(
                policy_consumption_unemployed[type_idx, :, :, period_idx],
                (
                    policy_consumption_unemployed[type_idx, :, -1, period_idx]
                    + consumption_diff
                )[..., np.newaxis],
                axis=1,
            ),
            hc_simulated[type_idx, :],
            assets_simulated[type_idx, :],
            method=interpolation_method,
        )

        consumption_unemployed_loss_simulated[
            type_idx, :
        ] = interpolate_2d_ordered_to_unordered(
            hc_grid_reduced_e_a1,
            assets_grid_e_a1,
            np.append(
                policy_consumption_unemployed_loss[type_idx, :, :, period_idx],
                (
                    policy_consumption_unemployed_loss[type_idx, :, -1, period_idx]
                    + consumption_diff
                )[..., np.newaxis],
                axis=1,
            ),
            hc_simulated[type_idx, :],
            assets_simulated[type_idx, :],
            method=interpolation_method,
        )

        consumption_employed_simulated[
            type_idx, :
        ] = interpolate_2d_ordered_to_unordered(
            hc_grid_reduced_e_a1,
            assets_grid_e_a1,
            np.append(
                policy_consumption_employed[type_idx, :, :, period_idx],
                (
                    policy_consumption_employed[type_idx, :, -1, period_idx]
                    + consumption_diff
                )[..., np.newaxis],
                axis=1,
            ),
            hc_simulated[type_idx, :],
            assets_simulated[type_idx, :],
            method=interpolation_method,
        )

    # construct combined array of simulated consumption levels
    consumption_simulated = (
        consumption_employed_simulated * employed_simulated
        + consumption_unemployed_simulated * unemployed_simulated
        + consumption_unemployed_loss_simulated * unemployed_loss_simulated
    )

    if np.any(np.isnan(consumption_simulated)):
        warnings.warn("NaN values in simulated consumption levels")

    return consumption_simulated


def _simulate_savings(
    employed_simulated,
    nonemployed_simulated,
    consumption_simulated,
    assets_simulated,
    wage_hc_factor_simulated,
    income_tax_rate_vector,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # compute next period asset holdings via bc
    savings_employed_simulated = (
        assets_simulated * (1 + interest_rate)
        + np.repeat((1 - income_tax_rate_vector[:, period_idx]), n_simulations).reshape(
            (n_types, n_simulations)
        )
        * wage_level
        * wage_hc_factor_simulated
        - consumption_simulated
    ) * employed_simulated
    savings_nonemployed_simulated = (
        assets_simulated * (1 + interest_rate)
        + simulate_ui_benefits(
            wage_level * wage_hc_factor_simulated,
            ui_replacement_rate_vector,
            ui_floor,
            ui_cap,
            period_idx,
        )
        - consumption_simulated
    ) * nonemployed_simulated

    # construct combined array of simulated consumption levels
    savings_simulated = savings_employed_simulated + savings_nonemployed_simulated

    # run some checks on savings, then return
    if np.any(np.isnan(savings_simulated)):
        warnings.warn("NaN values in simulated savings")

    if np.any(savings_simulated < assets_min):
        warnings.warn(
            "simulated savings below lower bound of asset grid; adjusting savings."
        )
        savings_simulated = np.maximum(savings_simulated, assets_min + eps)

    if np.any(savings_simulated > assets_max):
        warnings.warn(
            "simulated savings above upper bound of asset grid; adjusting savings."
        )
        savings_simulated = np.minimum(savings_simulated, assets_max - eps)

    return savings_simulated


def _get_statistics_consumption_phase(
    employed_simulated,
    unemployed_simulated,
    unemployed_loss_simulated,
    nonemployed_simulated,
    consumption_simulated,
    wage_hc_factor_simulated,
    wage_hc_factor_pre_displacement_simulated,
    duration_unemployed_simulated,
    duration_since_displacement_simulated,
    hc_simulated,
    assets_simulated,
    income_tax_rate_vector,
    ui_replacement_rate_vector,
    ui_floor,
    ui_cap,
    period_idx,
):

    # labor force status statistics
    share_employed = np.mean(employed_simulated, axis=1)
    share_unemployed = np.mean(unemployed_simulated, axis=1)
    share_unemployed_loss = np.mean(unemployed_loss_simulated, axis=1)
    share_nonemployed = np.mean(nonemployed_simulated, axis=1)

    # consumption statistics
    log_consumption_employed_mean = conditional_mean(
        np.log(consumption_simulated), employed_simulated, axis=1
    )
    log_consumption_nonemployed_mean = conditional_mean(
        np.log(consumption_simulated), nonemployed_simulated, axis=1
    )
    consumption_employed_mean = conditional_mean(
        consumption_simulated, employed_simulated, axis=1
    )
    consumption_nonemployed_mean = conditional_mean(
        consumption_simulated, nonemployed_simulated, axis=1
    )
    # consumption_nonemployed_stats = np.array(
    #     [
    #         np.mean(consumption_nonemployed_simulated[nonemployed_simulated]),
    #         np.median(consumption_nonemployed_simulated[nonemployed_simulated]),
    #         np.min(consumption_nonemployed_simulated[nonemployed_simulated]),
    #         np.std(consumption_nonemployed_simulated[nonemployed_simulated]),
    #     ]
    # )

    # utility statistics
    marginal_utility_nonemployed_mean = conditional_mean(
        _consumption_utility_dx(consumption_simulated), nonemployed_simulated, axis=1
    )

    # hc statistics
    hc_mean = np.mean(hc_simulated, axis=1)
    hc_employed_mean = conditional_mean(hc_simulated, employed_simulated, axis=1)
    hc_nonemployed_mean = conditional_mean(hc_simulated, nonemployed_simulated, axis=1)

    # wage statistics
    wage_hc_factor_mean = np.mean(wage_hc_factor_simulated, axis=1)
    wage_hc_factor_employed_mean = conditional_mean(
        wage_hc_factor_simulated, employed_simulated, axis=1
    )
    wage_hc_factor_unemployed_loss_mean = conditional_mean(
        wage_hc_factor_simulated, unemployed_loss_simulated, axis=1
    )
    wage_hc_factor_nonemployed_mean = conditional_mean(
        wage_hc_factor_simulated, nonemployed_simulated, axis=1
    )
    wage_hc_factor_displaced_mean = np.full((n_types, 6), np.nan)
    wage_hc_factor_nondisplaced_mean = np.full((n_types, 6), np.nan)
    for time in range(6):
        wage_hc_factor_displaced_mean[:, time] = conditional_mean(
            wage_hc_factor_pre_displacement_simulated,
            np.logical_and(
                nonemployed_simulated,
                duration_unemployed_simulated == time,
            ),
            axis=1,
        )
        wage_hc_factor_nondisplaced_mean[:, time] = conditional_mean(
            wage_hc_factor_simulated,
            np.logical_and(
                employed_simulated,
                duration_since_displacement_simulated == time,
            ),
            axis=1,
        )
    wage_hc_factor_pre_displacement_mean = conditional_mean(
        wage_hc_factor_pre_displacement_simulated, nonemployed_simulated, axis=1
    )

    # income statistics
    labor_income_simulated = (
        wage_level
        * wage_hc_factor_simulated
        * np.repeat((1 - income_tax_rate_vector[:, period_idx]), n_simulations).reshape(
            (n_types, n_simulations)
        )
        * employed_simulated
    )
    pre_unemployment_wage_simulated = (
        wage_level * wage_hc_factor_pre_displacement_simulated * nonemployed_simulated
    )
    ui_benefits_simulated = (
        simulate_ui_benefits(
            pre_unemployment_wage_simulated,
            ui_replacement_rate_vector,
            ui_floor,
            ui_cap,
            period_idx,
        )
        * nonemployed_simulated
    )
    income_simulated = labor_income_simulated + ui_benefits_simulated

    income_median = np.median(income_simulated)

    # UI statistics
    ui_benefits_mean = conditional_mean(
        ui_benefits_simulated, nonemployed_simulated, axis=1
    )
    ui_effective_replacement_rate = conditional_mean(
        ui_benefits_simulated, nonemployed_simulated, axis=1
    ) / conditional_mean(pre_unemployment_wage_simulated, nonemployed_simulated, axis=1)
    ui_share_floor_binding = conditional_mean(
        (ui_benefits_simulated == ui_floor), nonemployed_simulated, axis=1
    )
    ui_share_cap_binding = conditional_mean(
        (ui_benefits_simulated == ui_cap), nonemployed_simulated, axis=1
    )

    # wealth statistics
    assets_mean = np.mean(assets_simulated, axis=1)
    assets_nonemployed_mean = conditional_mean(
        assets_simulated, nonemployed_simulated, axis=1
    )
    assets_distribution = np.full((n_types, assets_grid_size), np.nan)
    distribution_hc_assets_nonemployed = np.full(
        (n_types, hc_grid_reduced_size, assets_grid_size), np.nan
    )
    for type_idx in range(n_types):
        assets_distribution[type_idx, :] = (
            np.histogram(
                a=assets_simulated[type_idx, :], bins=np.append(assets_grid, np.inf)
            )[0]
            / n_simulations
        )
        distribution_hc_assets_nonemployed[type_idx, :] = (
            np.histogram2d(
                x=np.squeeze(
                    hc_simulated[type_idx, nonemployed_simulated[type_idx, :]]
                ),
                y=np.squeeze(
                    assets_simulated[type_idx, nonemployed_simulated[type_idx, :]]
                ),
                bins=(
                    np.append(hc_grid_reduced, n_periods_working + 1),
                    np.append(assets_grid, np.inf),
                ),
            )[0]
            / np.sum(nonemployed_simulated[type_idx, :])
        )

    log_assets_over_income_nonemployed_mean = conditional_mean(
        np.log(assets_simulated / wage_hc_factor_simulated),
        np.logical_and(assets_simulated > 0, nonemployed_simulated),
        axis=1,
    )

    return (
        share_employed,
        share_unemployed,
        share_unemployed_loss,
        share_nonemployed,
        log_consumption_employed_mean,
        log_consumption_nonemployed_mean,
        consumption_employed_mean,
        consumption_nonemployed_mean,
        wage_hc_factor_mean,
        wage_hc_factor_employed_mean,
        wage_hc_factor_unemployed_loss_mean,
        wage_hc_factor_nonemployed_mean,
        wage_hc_factor_displaced_mean,
        wage_hc_factor_nondisplaced_mean,
        wage_hc_factor_pre_displacement_mean,
        marginal_utility_nonemployed_mean,
        income_median,
        hc_mean,
        hc_employed_mean,
        hc_nonemployed_mean,
        ui_benefits_mean,
        ui_effective_replacement_rate,
        ui_share_floor_binding,
        ui_share_cap_binding,
        assets_mean,
        assets_nonemployed_mean,
        assets_distribution,
        distribution_hc_assets_nonemployed,
        log_assets_over_income_nonemployed_mean,
    )


def _solve_and_simulate(controls, calibration):

    global assets_grid_size
    global assets_grid
    global assets_grid_e_a
    global assets_grid_e_a1
    global assets_grid_n_e_a
    global assets_grid_n_e_a1
    global assets_max
    global assets_min
    global contact_rate
    global borrowing_limit_e_a
    global discount_factor
    global eps
    global hc_max
    global hc_grid
    global hc_grid_reduced
    global hc_grid_reduced_e_a
    global hc_grid_reduced_e_a1
    global hc_grid_reduced_n_e_a
    global hc_grid_reduced_n_e_a1
    global hc_grid_reduced_size
    global interest_rate
    global interpolation_method
    global interpolation_method
    global job_finding_probability_grid
    global leisure_grid
    global leisure_utility
    global leisure_utility_dx
    global leisure_utility_dxdx
    global leisure_utility_dx_min
    global leisure_utility_dx_max
    global leisure_utility_on_search_grid
    global n_periods_working
    global n_periods_retired
    global n_simulations
    global n_types
    global risk_aversion_coefficient
    global search_effort_grid
    global search_effort_grid_size
    global wage_level

    # load controls
    eps = 0.0000000000001
    interpolation_method = controls["interpolation_method"]
    n_iter_solve_max = controls["n_iterations_solve_max"]
    n_simulations = controls["n_simulations"]
    seed_simulation = controls["seed_simulation"]
    show_progress = controls["show_progress_solve"]
    show_summary = controls["show_summary"]
    tolerance_solve = controls["tolerance_solve"]

    # load calibration
    assets_grid = np.array(calibration["assets_grid"])
    assets_max = calibration["assets_max"]
    assets_min = calibration["assets_min"]
    consumption_min = calibration["consumption_min"]
    consumption_tax_rate_init = np.array(
        calibration["consumption_tax_rate_init"][interpolation_method]
    )
    contact_rate = calibration["contact_rate"]
    discount_factor = calibration["discount_factor"]
    hc_grid_reduced = np.array(calibration["hc_grid_reduced"])
    hc_loss_probability = np.array(calibration["hc_loss_probability"])
    income_tax_rate_init = np.array(
        calibration["income_tax_rate_init"][interpolation_method]
    )
    instrument = calibration["instrument"]
    leisure_utility = np.array(calibration["leisure_utility"])
    leisure_utility_dx = np.array(calibration["leisure_utility_dx"])
    leisure_utility_dxdx = np.array(calibration["leisure_utility_dxdx"])
    n_periods_retired = calibration["n_periods_retired"]
    n_periods_working = calibration["n_periods_working"]
    n_types = calibration["n_types"]
    pension_benefits = np.array(calibration["pension_benefits"])
    risk_aversion_coefficient = calibration["risk_aversion_coefficient"]
    search_effort_grid_size = calibration["search_effort_grid_size"]
    search_effort_max = calibration["search_effort_max"]
    search_effort_min = calibration["search_effort_min"]
    separation_rate_vector = np.array(calibration["separation_rate_vector"])
    type_weights = np.array(calibration["type_weights"])
    ui_cap = calibration["ui_cap"]
    ui_floor = calibration["ui_floor"]
    ui_replacement_rate_vector = np.array(calibration["ui_replacement_rate_vector"])
    wage_hc_factor_vector = np.array(calibration["wage_hc_factor_vector"])
    wage_level = calibration["wage_level"]
    wage_loss_factor_vector = np.array(calibration["wage_loss_factor_vector"])
    wage_loss_reference_vector = np.array(calibration["wage_loss_reference_vector"])

    # calculate derived parameters
    assets_grid_size = len(assets_grid)
    hc_grid_reduced_size = len(hc_grid_reduced)
    hc_grid = np.arange(n_periods_working + 1)
    hc_max = np.amax(hc_grid)
    interest_rate = (1 - discount_factor) / discount_factor
    leisure_grid = np.linspace(
        search_effort_min, search_effort_max, len(leisure_utility)
    )
    search_effort_grid = np.linspace(
        search_effort_min, search_effort_max, search_effort_grid_size
    )
    leisure_utility_on_search_grid = leisure_utility_interpolated(search_effort_grid)
    leisure_utility_dx_max = leisure_utility_dx_interpolated(search_effort_max)
    leisure_utility_dx_min = leisure_utility_dx_interpolated(search_effort_min)
    job_finding_probability_grid = job_finding_probability(search_effort_grid)
    borrowing_limit_e_a = np.full((hc_grid_reduced_size, assets_grid_size), assets_min)
    n_types = separation_rate_vector.shape[0]

    # generate grids
    assets_grid_e_a = (
        np.repeat(assets_grid, hc_grid_reduced_size)
        .reshape(assets_grid_size, hc_grid_reduced_size)
        .T
    )
    assets_grid_e_a1 = np.append(
        assets_grid_e_a, np.full((hc_grid_reduced_size, 1), assets_max), axis=1
    )
    assets_grid_n_e_a = np.tile(assets_grid, n_types * hc_grid_reduced_size).reshape(
        (n_types, hc_grid_reduced_size, assets_grid_size)
    )
    assets_grid_n_e_a1 = np.append(
        assets_grid_n_e_a,
        np.full((n_types, hc_grid_reduced_size, 1), assets_max),
        axis=2,
    )

    hc_grid_reduced_e_a = np.repeat(hc_grid_reduced, assets_grid_size).reshape(
        hc_grid_reduced_size, assets_grid_size
    )
    hc_grid_reduced_e_a1 = np.append(
        hc_grid_reduced_e_a, hc_grid_reduced[..., np.newaxis], axis=1
    )
    hc_grid_reduced_n_e_a = np.tile(
        hc_grid_reduced, n_types * assets_grid_size
    ).reshape((n_types, assets_grid_size, hc_grid_reduced_size))
    hc_grid_reduced_n_e_a = np.moveaxis(hc_grid_reduced_n_e_a, 2, 1)

    hc_grid_reduced_n_e_a1 = np.append(
        hc_grid_reduced_n_e_a,
        np.tile(hc_grid_reduced, n_types).reshape((n_types, hc_grid_reduced_size, 1)),
        axis=2,
    )

    wage_hc_factor_grid = np.repeat(
        wage_hc_factor_vector[:, hc_grid_reduced],
        assets_grid_size,
    ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))

    # check instrument
    if instrument not in [
        "income_tax_rate",
        "income_tax_shift",
        "consumption_tax_rate",
    ]:
        raise ValueError(
            "error in equilibrium instrument; choose one of "
            "['income_tax_rate', 'income_tax_shift', 'consumption_tax_rate']"
        )

    # set initial values for policy rates
    consumption_tax_rate = consumption_tax_rate_init

    if income_tax_rate_init.shape == (n_types,):
        income_tax_rate = income_tax_rate_init
        income_tax_rate_vector = np.repeat(income_tax_rate, n_periods_working).reshape(
            (n_types, n_periods_working)
        )
    elif income_tax_rate_init.shape == (n_types, n_periods_working):
        income_tax_rate_vector = income_tax_rate_init
        income_tax_shift = 0.0
    else:
        raise ValueError("error in input variable income_tax_rate_init")

    if ui_cap == "None":
        ui_cap = np.Inf
    if ui_floor == "None":
        ui_floor = 0.0

    # initiate objects for iteration
    instrument_hist = []
    error = 100
    n_iter = 0

    #######################################################
    # SOLUTION

    while error > tolerance_solve and n_iter <= n_iter_solve_max:

        # Initialize objects

        # store current instrument rate
        if instrument == "income_tax_rate":
            instrument_hist += [copy.deepcopy(income_tax_rate)]
        elif instrument == "income_tax_shift":
            instrument_hist += [copy.deepcopy(income_tax_shift)]
        elif instrument == "consumption_tax_rate":
            instrument_hist += [copy.deepcopy(consumption_tax_rate)]

        # policy functions [type x hc x assets x age (working + first retirement period)]
        policy_consumption_unemployed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        policy_consumption_unemployed_loss = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        policy_consumption_employed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        policy_effort_searching = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        policy_effort_searching_loss = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )

        # value functions [type x hc x assets x age]
        value_unemployed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        value_unemployed_loss = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        value_employed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        value_searching = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        value_searching_loss = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )

        # cost functions [type x hc x assets x age]
        cost_unemployed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        cost_unemployed_loss = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )
        cost_employed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working + 1,
            ),
            np.nan,
        )

        # RETIREMENT PERIOD (discounted to first period of retirement)

        # retirement consumption levels [type x hc x assets]
        # (annuity income from assets + pension benefits - consumption tax)
        annuity_income = (
            assets_grid * interest_rate / (1 - discount_factor ** n_periods_retired)
        )
        annuity_income = np.tile(
            annuity_income, n_types * hc_grid_reduced_size
        ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))
        pension_income = np.repeat(
            pension_benefits, (hc_grid_reduced_size * assets_grid_size)
        ).reshape((n_types, hc_grid_reduced_size, assets_grid_size))

        policy_consumption_retired = np.repeat(
            1 / (1 + consumption_tax_rate), (hc_grid_reduced_size * assets_grid_size)
        ).reshape((n_types, hc_grid_reduced_size, assets_grid_size)) * (
            annuity_income + pension_income
        )

        policy_consumption_retired = np.maximum(
            policy_consumption_retired, consumption_min
        )

        # value of being retired [hc full x assets]
        value_retired = (
            (1 - discount_factor ** n_periods_retired)
            / (1 - discount_factor)
            * consumption_utility(policy_consumption_retired)
        )

        # cost to the government of paying pension benefits throughout retirement
        # by hc level (reduced) [type x hc x assets]
        cost_retired = (1 - discount_factor ** n_periods_retired) / (
            1 - discount_factor
        ) / np.repeat(
            (1 + consumption_tax_rate), (hc_grid_reduced_size * assets_grid_size)
        ).reshape(
            (n_types, hc_grid_reduced_size, assets_grid_size)
        ) * pension_income - 1 / discount_factor * np.repeat(
            consumption_tax_rate / (1 + consumption_tax_rate),
            (hc_grid_reduced_size * assets_grid_size),
        ).reshape(
            (n_types, hc_grid_reduced_size, assets_grid_size)
        ) * assets_grid_n_e_a

        # Starting in the last period of the working life
        period_idx = n_periods_working - 1  # zero-based indexing

        # store policy, value, and cost functions for first period of
        # retirement (hc loss materializes upon retirement)
        policy_consumption_employed[:, :, :, -1] = policy_consumption_retired
        policy_consumption_unemployed[:, :, :, -1] = policy_consumption_retired
        policy_consumption_unemployed_loss[
            :, :, :, -1
        ] = interpolate_n_h_a_ordered_to_unordered(
            hc_grid_reduced_n_e_a,
            assets_grid_n_e_a,
            policy_consumption_retired,
            _hc_after_loss_n_agents(
                hc_grid_reduced_n_e_a,
                wage_loss_factor_vector,
                wage_loss_reference_vector,
                period_idx + 1,
            ),
            assets_grid_n_e_a,
            method=interpolation_method,
        )

        # workers transition to "employed" status in retirement with certainty,
        # but without incurring disutility from searching; technical assumption
        # to facilitate coding, but without consequences for results
        policy_effort_searching[:, :, :, -1] = 1.0
        policy_effort_searching_loss[:, :, :, -1] = 1.0

        value_employed[:, :, :, -1] = value_retired
        value_unemployed[:, :, :, -1] = value_retired
        value_unemployed_loss[:, :, :, -1] = interpolate_n_h_a_ordered_to_unordered(
            hc_grid_reduced_n_e_a,
            assets_grid_n_e_a,
            value_retired,
            _hc_after_loss_n_agents(
                hc_grid_reduced_n_e_a,
                wage_loss_factor_vector,
                wage_loss_reference_vector,
                period_idx + 1,
            ),
            assets_grid_n_e_a,
            method=interpolation_method,
        )
        value_searching[:, :, :, -1] = value_retired
        value_searching_loss[:, :, :, -1] = interpolate_n_h_a_ordered_to_unordered(
            hc_grid_reduced_n_e_a,
            assets_grid_n_e_a,
            value_retired,
            _hc_after_loss_n_agents(
                hc_grid_reduced_n_e_a,
                wage_loss_factor_vector,
                wage_loss_reference_vector,
                period_idx + 1,
            ),
            assets_grid_n_e_a,
            method=interpolation_method,
        )

        cost_employed[:, :, :, -1] = cost_retired
        cost_unemployed[:, :, :, -1] = cost_retired
        cost_unemployed_loss[:, :, :, -1] = interpolate_n_h_a_ordered_to_unordered(
            hc_grid_reduced_n_e_a,
            assets_grid_n_e_a,
            cost_retired,
            _hc_after_loss_n_agents(
                hc_grid_reduced_n_e_a,
                wage_loss_factor_vector,
                wage_loss_reference_vector,
                period_idx + 1,
            ),
            assets_grid_n_e_a,
            method=interpolation_method,
        )

        # WORKING PERIOD (solving backwards for each t):
        while period_idx >= 0:
            for type_idx in range(n_types):

                # load policy, value, and cost functions for the next period
                policy_consumption_employed_next = policy_consumption_employed[
                    type_idx, :, :, period_idx + 1
                ]
                policy_consumption_unemployed_next = policy_consumption_unemployed[
                    type_idx, :, :, period_idx + 1
                ]
                policy_consumption_unemployed_loss_next = (
                    policy_consumption_unemployed_loss[type_idx, :, :, period_idx + 1]
                )
                policy_effort_searching_next = policy_effort_searching[
                    type_idx, :, :, period_idx + 1
                ]
                policy_effort_searching_loss_next = policy_effort_searching_loss[
                    type_idx, :, :, period_idx + 1
                ]

                value_employed_next = value_employed[type_idx, :, :, period_idx + 1]
                value_searching_next = value_searching[type_idx, :, :, period_idx + 1]
                value_searching_loss_next = value_searching_loss[
                    type_idx, :, :, period_idx + 1
                ]

                cost_employed_next = cost_employed[type_idx, :, :, period_idx + 1]
                cost_unemployed_next = cost_unemployed[type_idx, :, :, period_idx + 1]
                cost_unemployed_loss_next = cost_unemployed_loss[
                    type_idx, :, :, period_idx + 1
                ]

                # solve period
                (
                    policy_consumption_employed_now,
                    policy_consumption_unemployed_now,
                    policy_consumption_unemployed_loss_now,
                    policy_effort_searching_now,
                    policy_effort_searching_loss_now,
                    value_employed_now,
                    value_unemployed_now,
                    value_unemployed_loss_now,
                    value_searching_now,
                    value_searching_loss_now,
                    cost_employed_now,
                    cost_unemployed_now,
                    cost_unemployed_loss_now,
                ) = _solve_one_period(
                    policy_consumption_employed_next,
                    policy_consumption_unemployed_next,
                    policy_consumption_unemployed_loss_next,
                    policy_effort_searching_next,
                    policy_effort_searching_loss_next,
                    value_employed_next,
                    value_searching_next,
                    value_searching_loss_next,
                    cost_employed_next,
                    cost_unemployed_next,
                    cost_unemployed_loss_next,
                    hc_loss_probability[type_idx, ...],
                    separation_rate_vector[type_idx, ...],
                    wage_hc_factor_grid[type_idx, ...],
                    wage_hc_factor_vector[type_idx, ...],
                    wage_loss_factor_vector[type_idx, ...],
                    wage_loss_reference_vector[type_idx, ...],
                    consumption_tax_rate[type_idx, ...],
                    income_tax_rate_vector[type_idx, ...],
                    ui_replacement_rate_vector[type_idx, ...],
                    ui_floor,
                    ui_cap,
                    period_idx,
                )

                # store results
                policy_consumption_employed[
                    type_idx, :, :, period_idx
                ] = policy_consumption_employed_now
                policy_consumption_unemployed[
                    type_idx, :, :, period_idx
                ] = policy_consumption_unemployed_now
                policy_consumption_unemployed_loss[
                    type_idx, :, :, period_idx
                ] = policy_consumption_unemployed_loss_now
                policy_effort_searching[
                    type_idx, :, :, period_idx
                ] = policy_effort_searching_now
                policy_effort_searching_loss[
                    type_idx, :, :, period_idx
                ] = policy_effort_searching_loss_now

                value_employed[type_idx, :, :, period_idx] = value_employed_now
                value_unemployed[type_idx, :, :, period_idx] = value_unemployed_now
                value_unemployed_loss[
                    type_idx, :, :, period_idx
                ] = value_unemployed_loss_now
                value_searching[type_idx, :, :, period_idx] = value_searching_now
                value_searching_loss[
                    type_idx, :, :, period_idx
                ] = value_searching_loss_now

                cost_employed[type_idx, :, :, period_idx] = cost_employed_now
                cost_unemployed[type_idx, :, :, period_idx] = cost_unemployed_now
                cost_unemployed_loss[
                    type_idx, :, :, period_idx
                ] = cost_unemployed_loss_now

            # initiate next iteration
            period_idx -= 1

        # obtain aggregate measures

        pv_utility_computed = np.full(n_types, np.nan)
        pv_cost_computed = np.full(n_types, np.nan)
        for type_idx in range(n_types):
            # pv of utility of searcher with age=0, assets=0 and hc=0
            pv_utility_computed[type_idx] = interpolate_1d(
                assets_grid,
                value_searching[type_idx, 0, :, 0],
                0,
                method=interpolation_method,
            )

            # pv of net cost to government for searcher with age=0, assets=0 and hc=0
            search_effort_at_entry = interpolate_1d(
                assets_grid,
                policy_effort_searching[type_idx, 0, :, 0],
                0,
                method=interpolation_method,
            )
            pv_cost_computed[type_idx] = (1 - search_effort_at_entry) * interpolate_1d(
                assets_grid,
                cost_unemployed[type_idx, 0, :, 0],
                0,
                method=interpolation_method,
            ) + search_effort_at_entry * interpolate_1d(
                assets_grid,
                cost_employed[type_idx, 0, :, 0],
                0,
                method=interpolation_method,
            )

        # correct for unbalanced government budget
        pv_utility_corrected = pv_utility_computed - 0.55 * pv_cost_computed

        # average over types
        average_pv_utility_computed = np.average(
            pv_utility_computed, weights=type_weights
        )
        average_pv_cost_computed = np.average(pv_cost_computed, weights=type_weights)
        average_pv_utility_computed_corrected = np.average(
            pv_utility_corrected, weights=type_weights
        )

        # print output summary
        if show_summary:
            summary_solve = np.array(
                (
                    ("n_iter", n_iter),
                    ("cost at entry", average_pv_cost_computed),
                    ("value at entry", np.round(average_pv_utility_computed, 5)),
                    (
                        "welfare corrected",
                        np.round(average_pv_utility_computed_corrected, 5),
                    ),
                    (f"initial {instrument}", np.round(instrument_hist[0], 5)),
                    (f"current {instrument}", np.round(instrument_hist[-1], 5)),
                )
            )
            print(summary_solve)

        error = abs(average_pv_cost_computed)

        # prepare next outer iteration of solution algorithm
        if error <= tolerance_solve or n_iter == n_iter_solve_max:
            break  # don't update tax rate in output iteration
        else:
            # update iteration counter
            n_iter += 1

            # update income tax rate
            adjustment_factor = 0.0079 * n_iter ** (-0.25)

            if instrument == "income_tax_rate":
                income_tax_rate += adjustment_factor * average_pv_cost_computed
                income_tax_rate_vector = np.repeat(
                    income_tax_rate, n_periods_working
                ).reshape((n_types, n_periods_working))
            elif instrument == "income_tax_shift":
                income_tax_shift += adjustment_factor * average_pv_cost_computed
                income_tax_rate_vector = income_tax_rate_init + income_tax_shift
            elif instrument == "consumption_tax_rate":
                consumption_tax_rate += adjustment_factor * average_pv_cost_computed

    if show_progress:
        print("end solution")

    ########################################################
    # SIMULATION
    if controls["run_simulation"]:

        # I: INITIALISATION

        # set seed
        np.random.seed(seed_simulation)

        # (a) initiate objects for simulation

        # booleans for worker status
        searching_loss_simulated = np.zeros((n_types, n_simulations), dtype=bool)
        searching_simulated = np.ones((n_types, n_simulations), dtype=bool)
        employed_simulated = np.zeros((n_types, n_simulations), dtype=bool)
        searching_all_simulated = (
            searching_simulated + searching_loss_simulated
        ).astype(bool)

        # assets tracker
        assets_simulated = np.zeros((n_types, n_simulations))

        # unemployment duration trackers
        duration_unemployed_simulated = np.zeros((n_types, n_simulations))
        duration_since_displacement_simulated = np.zeros((n_types, n_simulations))

        # human capital tracker
        hc_simulated = np.zeros((n_types, n_simulations))
        hc_pre_displacement_simulated = np.zeros((n_types, n_simulations))

        # value tracker
        pv_utility_simulated = np.zeros((n_types, n_simulations))

        # (c) initiate objects for statistics
        # aggregate statistics
        discount_factor_compounded = 1
        pv_government_income = np.zeros(n_types)
        pv_government_cost = np.zeros(n_types)
        total_benefits = np.zeros(n_types)

        # government budget statistics
        net_government_spending_working = np.full((n_types, n_periods_working), np.nan)

        # hc statistics
        hc_mean = np.full((n_types, n_periods_working), np.nan)
        hc_employed_mean = np.full((n_types, n_periods_working), np.nan)
        hc_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)

        # wage and income statistics
        income_median = np.full((n_types, n_periods_working), np.nan)
        wage_hc_factor_mean = np.full((n_types, n_periods_working), np.nan)
        wage_hc_factor_employed_mean = np.full((n_types, n_periods_working), np.nan)
        wage_hc_factor_unemployed_loss_mean = np.full(
            (n_types, n_periods_working), np.nan
        )
        wage_hc_factor_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)
        wage_hc_factor_displaced_mean = np.full((n_types, 6, n_periods_working), np.nan)
        wage_hc_factor_nondisplaced_mean = np.full(
            (n_types, 6, n_periods_working), np.nan
        )
        wage_hc_factor_pre_displacement_mean = np.full(
            (n_types, n_periods_working), np.nan
        )

        # ui benefit statistics
        ui_benefits_mean = np.full((n_types, n_periods_working), np.nan)
        ui_effective_replacement_rate = np.full((n_types, n_periods_working), np.nan)
        ui_share_floor_binding = np.full((n_types, n_periods_working), np.nan)
        ui_share_cap_binding = np.full((n_types, n_periods_working), np.nan)

        # wealth statistics
        assets_mean = np.full((n_types, n_periods_working), np.nan)
        assets_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)
        assets_distribution = np.full(
            (n_types, assets_grid_size, n_periods_working), np.nan
        )
        log_assets_over_income_nonemployed_mean = np.full(
            (n_types, n_periods_working), np.nan
        )

        # utility statistics
        marginal_utility_nonemployed_mean = np.full(
            (n_types, n_periods_working), np.nan
        )

        # labor force status statistics
        share_employed = np.full((n_types, n_periods_working), np.nan)
        share_unemployed = np.full((n_types, n_periods_working), np.nan)
        share_unemployed_loss = np.full((n_types, n_periods_working), np.nan)
        share_nonemployed = np.full((n_types, n_periods_working), np.nan)
        share_searching = np.full((n_types, n_periods_working), np.nan)

        # consumption statistics
        consumption_employed_mean = np.full((n_types, n_periods_working), np.nan)
        consumption_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)
        # consumption_nonemployed_stats = np.full((n_types, 4, n_periods_working), np.nan)
        log_consumption_employed_mean = np.full((n_types, n_periods_working), np.nan)
        log_consumption_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)

        # labor markets statistics
        job_finding_probability_unemployed_mean = np.full(
            (n_types, n_periods_working), np.nan
        )
        job_finding_probability_unemployed_loss_mean = np.full(
            (n_types, n_periods_working), np.nan
        )
        job_finding_probability_nonemployed_mean = np.full(
            (n_types, n_periods_working), np.nan
        )
        job_finding_rate_searching_mean = np.full((n_types, n_periods_working), np.nan)
        job_finding_rate_searching_loss_mean = np.full(
            (n_types, n_periods_working), np.nan
        )
        job_finding_rate_searching_all_mean = np.full(
            (n_types, n_periods_working), np.nan
        )

        duration_unemployed_weeks_mean = np.full((n_types, n_periods_working), np.nan)
        duration_unemployed_median = np.full((n_types, n_periods_working), np.nan)
        duration_unemployed_stdev = np.full((n_types, n_periods_working), np.nan)

        wage_loss_median = np.full((n_types, n_periods_working), np.nan)

        # cross sectional statistics
        distribution_hc_assets_nonemployed = np.full(
            (
                n_types,
                hc_grid_reduced_size,
                assets_grid_size,
                n_periods_working,
            ),
            np.nan,
        )

        # II: SIMULATION

        # (a) simulation from 1 to end of working life
        for period_idx in range(n_periods_working):

            # (i) search phase

            # simulate search effort
            # effort_searching_simulated = interpolate_n_e_a_ordered_to_unordered(
            #     hc_grid_reduced_n_e_a,
            #     assets_grid_n_e_a,
            #     policy_effort_searching[:, :, :, period_idx],
            #     hc_simulated,
            #     assets_simulated,
            # )
            # effort_searching_loss_simulated = interpolate_n_e_a_ordered_to_unordered(
            #     hc_grid_reduced_n_e_a,
            #     assets_grid_n_e_a,
            #     policy_effort_searching_loss[:, :, :, period_idx],
            #     hc_simulated,
            #     assets_simulated,
            # )
            effort_searching_simulated = np.full((n_types, n_simulations), np.nan)
            effort_searching_loss_simulated = np.full((n_types, n_simulations), np.nan)
            for type_idx in range(n_types):
                effort_searching_simulated[
                    type_idx, :
                ] = interpolate_2d_ordered_to_unordered(
                    hc_grid_reduced_e_a,
                    assets_grid_e_a,
                    policy_effort_searching[type_idx, :, :, period_idx],
                    hc_simulated[type_idx, :],
                    assets_simulated[type_idx, :],
                )
                effort_searching_loss_simulated[
                    type_idx, :
                ] = interpolate_2d_ordered_to_unordered(
                    hc_grid_reduced_e_a,
                    assets_grid_e_a,
                    policy_effort_searching_loss[type_idx, :, :, period_idx],
                    hc_simulated[type_idx, :],
                    assets_simulated[type_idx, :],
                )

            effort_searching_simulated = np.minimum(
                np.maximum(effort_searching_simulated, 0.0), 1.0
            )
            effort_searching_loss_simulated = np.minimum(
                np.maximum(effort_searching_loss_simulated, 0.0), 1.0
            )

            job_finding_probability_searching_simulated = job_finding_probability(
                effort_searching_simulated
            )
            job_finding_probability_searching_loss_simulated = job_finding_probability(
                effort_searching_loss_simulated
            )

            # compute search phase statistics
            share_searching[:, period_idx] = np.mean(searching_all_simulated, axis=1)
            job_finding_probability_nonemployed_mean[:, period_idx] = conditional_mean(
                (
                    job_finding_probability_searching_simulated * searching_simulated
                    + job_finding_probability_searching_loss_simulated
                    * searching_loss_simulated
                ),
                searching_all_simulated,
                axis=1,
            )
            job_finding_probability_unemployed_mean[:, period_idx] = conditional_mean(
                job_finding_probability_searching_simulated, searching_simulated, axis=1
            )
            job_finding_probability_unemployed_loss_mean[
                :, period_idx
            ] = conditional_mean(
                job_finding_probability_searching_loss_simulated,
                searching_loss_simulated,
                axis=1,
            )

            # generate transition events
            job_finding_event_searching_simulated = (
                job_finding_probability_searching_simulated
                >= np.random.rand(n_types, n_simulations)
            ).astype(bool)
            job_finding_event_searching_loss_simulated = (
                job_finding_probability_searching_loss_simulated
                >= np.random.rand(n_types, n_simulations)
            ).astype(bool)

            # calculate average job finding rates
            job_finding_rate_searching_all_mean[:, period_idx] = conditional_mean(
                (
                    job_finding_event_searching_simulated * searching_simulated
                    + job_finding_event_searching_loss_simulated
                    * searching_loss_simulated
                ),
                searching_all_simulated,
                axis=1,
            )
            job_finding_rate_searching_mean[:, period_idx] = conditional_mean(
                job_finding_event_searching_simulated, searching_simulated, axis=1
            )
            job_finding_rate_searching_loss_mean[:, period_idx] = conditional_mean(
                job_finding_event_searching_loss_simulated,
                searching_loss_simulated,
                axis=1,
            )

            # calculate unemployment duration statistics  # todo: check timing
            duration_unemployed_simulated_weeks = _get_duration_weeks(
                (
                    job_finding_probability_searching_simulated * searching_simulated
                    + job_finding_probability_searching_loss_simulated
                    * searching_loss_simulated
                ),
                duration_unemployed_simulated,
            )
            duration_unemployed_weeks_mean[:, period_idx] = conditional_mean(
                np.minimum(
                    duration_unemployed_simulated_weeks, 98
                ),  # unemployment duration is capped at 98 weeks in the data
                searching_all_simulated,
                axis=1,
            )
            duration_unemployed_median[:, period_idx] = [
                np.median(
                    duration_unemployed_simulated[i, searching_all_simulated[i, :]]
                )
                for i in range(n_types)
            ]
            duration_unemployed_stdev[:, period_idx] = [
                np.std(duration_unemployed_simulated[i, searching_all_simulated[i, :]])
                for i in range(n_types)
            ]

            # simulate transitions from search phase to consumption phase

            # transitions of labor force status
            employed_simulated = (
                employed_simulated
                + searching_simulated * job_finding_event_searching_simulated
                + searching_loss_simulated * job_finding_event_searching_loss_simulated
            ).astype(bool)
            unemployed_simulated = (
                searching_simulated * (1 - job_finding_event_searching_simulated)
            ).astype(bool)
            unemployed_loss_simulated = (
                searching_loss_simulated
                * (1 - job_finding_event_searching_loss_simulated)
            ).astype(bool)
            nonemployed_simulated = (
                unemployed_simulated + unemployed_loss_simulated
            ).astype(bool)

            # simulate hc transition to consumption phase
            hc_loss_simulated = np.full((n_types, n_simulations), np.nan)
            pre_displacement_wage_simulated = np.full((n_types, n_simulations), np.nan)
            new_wage_simulated = np.full((n_types, n_simulations), np.nan)
            for type_idx in range(n_types):
                hc_loss_simulated[type_idx, :] = (
                    (
                        hc_simulated[type_idx, :]
                        - _hc_after_loss_1_agent(
                            hc_simulated[type_idx, :],
                            wage_loss_factor_vector[type_idx, :],
                            wage_loss_reference_vector[type_idx, :],
                            period_idx,
                        )
                    )
                    * searching_loss_simulated[type_idx, :]
                    * job_finding_event_searching_loss_simulated[type_idx, :]
                )
                pre_displacement_wage_simulated[type_idx, :] = (
                    wage_level
                    * wage_hc_factor_interpolated_1_agent(
                        hc_simulated[type_idx, :], wage_hc_factor_vector[type_idx, :]
                    )
                    * searching_loss_simulated[type_idx, :]
                    * job_finding_event_searching_loss_simulated[type_idx, :]
                )
                new_wage_simulated[type_idx, :] = (
                    wage_level
                    * wage_hc_factor_interpolated_1_agent(
                        hc_simulated[type_idx, :] - hc_loss_simulated[type_idx, :],
                        wage_hc_factor_vector[type_idx, :],
                    )
                    * searching_loss_simulated[type_idx, :]
                    * job_finding_event_searching_loss_simulated[type_idx, :]
                )

            # calculate wage loss statistics
            wage_loss_simulated = new_wage_simulated - pre_displacement_wage_simulated
            wage_loss_median[:, period_idx] = np.array(
                [
                    np.median(
                        wage_loss_simulated[
                            type_idx,
                            searching_loss_simulated[type_idx, :]
                            * job_finding_event_searching_loss_simulated[type_idx, :],
                        ]
                    )
                    for type_idx in range(n_types)
                ]
            )

            # simulate hc loss upon reemployment
            hc_simulated = hc_simulated - hc_loss_simulated

            # transition of durations
            duration_unemployed_simulated = (
                duration_unemployed_simulated
                + searching_simulated * (1 - job_finding_event_searching_simulated)
                + searching_loss_simulated
                * (1 - job_finding_event_searching_loss_simulated)
            )  # +1 for workers that remain unemployed
            duration_unemployed_simulated = (
                duration_unemployed_simulated * nonemployed_simulated
            )  # =0 for everyone else

            # check for error in state simulation
            if (
                np.sum(
                    unemployed_simulated
                    + unemployed_loss_simulated
                    + employed_simulated
                )
                < n_simulations
            ):
                warnings.warn(
                    f"ERROR! in transition from search phase "
                    f"to consumption phase in period {period_idx}"
                )

            # (ii) consumption phase

            # simulate consumption
            consumption_simulated = _simulate_consumption(
                policy_consumption_employed,
                policy_consumption_unemployed,
                policy_consumption_unemployed_loss,
                employed_simulated,
                unemployed_simulated,
                unemployed_loss_simulated,
                hc_simulated,
                assets_simulated,
                period_idx,
            )

            # update wages
            wage_hc_factor_simulated = np.array(
                [
                    wage_hc_factor_interpolated_1_agent(
                        hc_simulated[i, :], wage_hc_factor_vector[i, :]
                    )
                    for i in range(n_types)
                ]
            )
            wage_hc_factor_pre_displacement_simulated = np.array(
                [
                    wage_hc_factor_interpolated_1_agent(
                        hc_pre_displacement_simulated[i, :], wage_hc_factor_vector[i, :]
                    )
                    for i in range(n_types)
                ]
            )

            # simulate savings
            savings_simulated = _simulate_savings(
                employed_simulated,
                nonemployed_simulated,
                consumption_simulated,
                assets_simulated,
                wage_hc_factor_simulated,
                income_tax_rate_vector,
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            )

            # compute consumption phase statistics

            # update aggregate variables
            income_tax_revenue_simulated = (
                np.repeat(income_tax_rate_vector[:, period_idx], n_simulations).reshape(
                    (n_types, n_simulations)
                )
                * wage_level
                * wage_hc_factor_simulated
                * employed_simulated
            )
            consumption_tax_revenue_simulated = (
                np.repeat(consumption_tax_rate, n_simulations).reshape(
                    n_types, n_simulations
                )
                * consumption_simulated
            )
            ui_benefits_simulated = (
                simulate_ui_benefits(
                    wage_level * wage_hc_factor_pre_displacement_simulated,
                    ui_replacement_rate_vector,
                    ui_floor,
                    ui_cap,
                    period_idx,
                )
                * nonemployed_simulated
            )

            pv_government_income += discount_factor_compounded * np.sum(
                income_tax_revenue_simulated, axis=1
            ) + discount_factor_compounded * np.sum(
                consumption_tax_revenue_simulated, axis=1
            )
            pv_government_cost += discount_factor_compounded * np.sum(
                ui_benefits_simulated, axis=1
            )
            total_benefits += np.sum(ui_benefits_simulated, axis=1)
            net_government_spending_working[:, period_idx] = (
                np.sum(income_tax_revenue_simulated, axis=1)
                + np.sum(consumption_tax_revenue_simulated, axis=1)
                - np.sum(ui_benefits_simulated, axis=1)
            )

            # get statistics
            (
                share_employed[:, period_idx],
                share_unemployed[:, period_idx],
                share_unemployed_loss[:, period_idx],
                share_nonemployed[:, period_idx],
                log_consumption_employed_mean[:, period_idx],
                log_consumption_nonemployed_mean[:, period_idx],
                consumption_employed_mean[:, period_idx],
                consumption_nonemployed_mean[:, period_idx],
                wage_hc_factor_mean[:, period_idx],
                wage_hc_factor_employed_mean[:, period_idx],
                wage_hc_factor_unemployed_loss_mean[:, period_idx],
                wage_hc_factor_nonemployed_mean[:, period_idx],
                wage_hc_factor_displaced_mean[:, :, period_idx],
                wage_hc_factor_nondisplaced_mean[:, :, period_idx],
                wage_hc_factor_pre_displacement_mean[:, period_idx],
                marginal_utility_nonemployed_mean[:, period_idx],
                income_median[:, period_idx],
                hc_mean[:, period_idx],
                hc_employed_mean[:, period_idx],
                hc_nonemployed_mean[:, period_idx],
                ui_benefits_mean[:, period_idx],
                ui_effective_replacement_rate[:, period_idx],
                ui_share_floor_binding[:, period_idx],
                ui_share_cap_binding[:, period_idx],
                assets_mean[:, period_idx],
                assets_nonemployed_mean[:, period_idx],
                assets_distribution[:, :, period_idx],
                distribution_hc_assets_nonemployed[:, :, :, period_idx],
                log_assets_over_income_nonemployed_mean[:, period_idx],
            ) = _get_statistics_consumption_phase(
                employed_simulated,
                unemployed_simulated,
                unemployed_loss_simulated,
                nonemployed_simulated,
                consumption_simulated,
                wage_hc_factor_simulated,
                wage_hc_factor_pre_displacement_simulated,
                duration_unemployed_simulated,
                duration_since_displacement_simulated,
                hc_simulated,
                assets_simulated,
                income_tax_rate_vector,
                ui_replacement_rate_vector,
                ui_floor,
                ui_cap,
                period_idx,
            )

            # update simulated discounted value
            pv_utility_simulated[
                searching_all_simulated
            ] += discount_factor_compounded * leisure_utility_interpolated(
                effort_searching_simulated[searching_all_simulated]
            )
            pv_utility_simulated[
                employed_simulated
            ] += discount_factor_compounded * consumption_utility(
                consumption_simulated[employed_simulated]
            )
            pv_utility_simulated[
                nonemployed_simulated
            ] += discount_factor_compounded * consumption_utility(
                np.maximum(
                    consumption_simulated[nonemployed_simulated],
                    consumption_min,
                )
            )

            if np.any(np.isnan(pv_utility_simulated)):
                warnings.warn("NaN in simulated discounted value at birth")

            # simulate transition

            # simulate transition events
            hc_loss_event_simulated = (
                np.repeat(hc_loss_probability, n_simulations).reshape(
                    n_types, n_simulations
                )
                >= np.random.rand(n_types, n_simulations)
            ).astype(bool)
            job_loss_event_simulated = (
                np.repeat(separation_rate_vector[:, period_idx], n_simulations).reshape(
                    n_types, n_simulations
                )
                >= np.random.rand(n_types, n_simulations)
            ).astype(bool)

            # simulate experience transition
            hc_simulated = (
                hc_simulated
                + np.full((n_types, n_simulations), 1.0) * employed_simulated
            )  # increase experience of employed workers by 1
            hc_pre_displacement_simulated = (
                hc_pre_displacement_simulated * nonemployed_simulated
                + hc_simulated * employed_simulated
            )

            # update duration tracker  # todo: resolve this abomination
            duration_unemployed_simulated = (
                duration_unemployed_simulated
                + np.logical_and(
                    duration_unemployed_simulated >= 1,
                    duration_unemployed_simulated <= 5,
                )
            )
            duration_unemployed_simulated = (
                duration_unemployed_simulated
                - 10
                * (duration_unemployed_simulated > 0)
                * employed_simulated
                * job_loss_event_simulated
            )
            duration_unemployed_simulated = (
                duration_unemployed_simulated
                + (duration_unemployed_simulated == 0)
                * employed_simulated
                * job_loss_event_simulated
            )
            duration_unemployed_simulated = duration_unemployed_simulated - 6 * (
                duration_unemployed_simulated > 5
            )
            duration_unemployed_simulated = np.maximum(duration_unemployed_simulated, 0)

            duration_since_displacement_simulated += (
                np.full((n_types, n_simulations), 1.0)
                * employed_simulated
                * (1 - job_loss_event_simulated)
            )  # +1 for all workers that are still employed
            duration_since_displacement_simulated *= (
                np.full((n_types, n_simulations), 1.0)
                * employed_simulated
                * (1 - job_loss_event_simulated)
            )  # =0 for everyone else
            duration_since_displacement_simulated = np.minimum(
                duration_since_displacement_simulated, 5
            )  # capped at 5

            # simulate transitions in employment status
            searching_simulated = (
                unemployed_simulated * (1 - hc_loss_event_simulated)
                + employed_simulated * job_loss_event_simulated
            ).astype(bool)
            searching_loss_simulated = (
                unemployed_loss_simulated
                + unemployed_simulated * hc_loss_event_simulated
            ).astype(bool)
            searching_all_simulated = (
                searching_simulated + searching_loss_simulated
            ).astype(bool)
            employed_simulated = (
                employed_simulated * (1 - job_loss_event_simulated)
            ).astype(bool)

            # update assets
            assets_simulated = savings_simulated

            # compound discount factor
            discount_factor_compounded = discount_factor_compounded * discount_factor

            # check for error in state simulation
            if (
                np.sum(
                    searching_simulated + searching_loss_simulated + employed_simulated
                )
                < n_simulations
            ):
                warnings.warn(
                    f"ERROR! in transition from consumption phase "
                    f"in period {period_idx} to search phase in {period_idx + 1}"
                )

        # retirement period

        # simulate one more transition to consumption phase
        unemployed_simulated = searching_simulated
        unemployed_loss_simulated = searching_loss_simulated

        # hc loss materialises upon retirement
        hc_loss_simulated = np.full((n_types, n_simulations), np.nan)
        for type_idx in range(n_types):
            hc_loss_simulated[type_idx, :] = (
                hc_simulated[type_idx, :]
                - _hc_after_loss_1_agent(
                    hc_simulated[type_idx, :],
                    wage_loss_factor_vector[type_idx, :],
                    wage_loss_reference_vector[type_idx, :],
                    period_idx + 1,
                )
            ) * searching_loss_simulated[type_idx, :]
        hc_simulated = (
            hc_simulated - hc_loss_simulated
        )  # hc loss materializes upon reemployment

        # interpolate consumption policies in first period of retirement
        consumption_employed_retired_simulated = np.full(
            (n_types, n_simulations), np.nan
        )
        consumption_unemployed_retired_simulated = np.full(
            (n_types, n_simulations), np.nan
        )
        consumption_unemployed_loss_retired_simulated = np.full(
            (n_types, n_simulations), np.nan
        )

        for type_idx in range(n_types):
            consumption_employed_retired_simulated[type_idx, :] = (
                interpolate_2d_ordered_to_unordered(
                    hc_grid_reduced_e_a,
                    assets_grid_e_a,
                    policy_consumption_employed[type_idx, :, :, period_idx + 1],
                    hc_simulated[type_idx, :],
                    assets_simulated[type_idx, :],
                    method=interpolation_method,
                )
                * employed_simulated[type_idx, :]
            )
            consumption_unemployed_retired_simulated[type_idx, :] = (
                interpolate_2d_ordered_to_unordered(
                    hc_grid_reduced_e_a,
                    assets_grid_e_a,
                    policy_consumption_unemployed[type_idx, :, :, period_idx + 1],
                    hc_simulated[type_idx, :],
                    assets_simulated[type_idx, :],
                    method=interpolation_method,
                )
                * unemployed_simulated[type_idx, :]
            )
            consumption_unemployed_loss_retired_simulated[type_idx, :] = (
                interpolate_2d_ordered_to_unordered(
                    hc_grid_reduced_e_a,
                    assets_grid_e_a,
                    policy_consumption_unemployed_loss[type_idx, :, :, period_idx + 1],
                    hc_simulated[type_idx, :],
                    assets_simulated[type_idx, :],
                    method=interpolation_method,
                )
                * unemployed_loss_simulated[type_idx, :]
            )

        consumption_retired_simulated = (
            consumption_employed_retired_simulated * employed_simulated
            + consumption_unemployed_retired_simulated * unemployed_simulated
            + consumption_unemployed_loss_retired_simulated * unemployed_loss_simulated
        )

        # compute discounted simulated retirement value
        discount_factor_retirement = (1 - discount_factor ** n_periods_retired) / (
            1 - discount_factor
        )
        value_retired_simulated = discount_factor_retirement * consumption_utility(
            consumption_retired_simulated
        )

        government_income_retired_simulated = (
            discount_factor_retirement
            * np.repeat(consumption_tax_rate, n_simulations).reshape(
                (n_types, n_simulations)
            )
            * consumption_retired_simulated
        )

        # add discounted simulated retirement value to
        # simulated expected discounted value at birth
        # and expected discounted government cost
        pv_utility_simulated += discount_factor_compounded * value_retired_simulated
        pv_government_income += discount_factor_compounded * np.sum(
            government_income_retired_simulated, axis=1
        )

        # average over simulations
        pv_utility_simulated = np.mean(pv_utility_simulated, axis=1)

        if show_progress:
            print("end simulation")
        #########################################################
        # (b) compute outcomes and store results

        # (i) compute outcomes
        pv_pensions = (
            discount_factor_compounded
            * (1 - discount_factor ** n_periods_retired)
            / (1 - discount_factor)
            * pension_benefits
            * n_simulations
        )
        pv_government_spending = pv_government_income - pv_government_cost - pv_pensions
        net_government_spending_all = np.append(
            net_government_spending_working,
            np.repeat(-pension_benefits * n_simulations, n_periods_retired).reshape(
                (n_types, n_periods_retired)
            ),
            axis=1,
        )

        # average over types
        average_pv_utility_simulated = np.average(
            pv_utility_simulated, weights=type_weights
        )
        average_pv_government_spending = np.average(
            pv_government_spending, weights=type_weights
        )

        diff_pv_utility = abs(
            average_pv_utility_simulated - average_pv_utility_computed
        )

        average_pv_utility_simulated_corrected = -(
            average_pv_utility_computed - 0 * abs(average_pv_government_spending)
        )

        if show_summary:
            summary_simulate = np.array(
                (
                    ("number of simulations", n_simulations),
                    ("value at entry (mean simulated)", average_pv_utility_simulated),
                    ("difference in value at entry", diff_pv_utility),
                    (
                        "net gov't budget (simulated)",
                        np.round(average_pv_government_spending, 3),
                    ),
                )
            )
            print(summary_simulate)

        # (ii) store some results
        out = {
            "assets_mean": assets_mean,
            "assets_distribution": assets_distribution,
            "assets_nonemployed_mean": assets_nonemployed_mean,
            "average_pv_cost_computed": average_pv_cost_computed,
            "average_pv_government_spending": average_pv_government_spending,
            "average_pv_utility_computed": average_pv_utility_computed,
            "average_pv_utility_computed_corrected": average_pv_utility_computed_corrected,
            "average_pv_utility_simulated": average_pv_utility_simulated,
            "average_pv_utility_simulated_corrected": average_pv_utility_simulated_corrected,
            "consumption_employed_mean": consumption_employed_mean,
            "consumption_nonemployed_mean": consumption_nonemployed_mean,
            "distribution_assets_hc_nonemployed": distribution_hc_assets_nonemployed,
            "duration_unemployed_weeks_mean": duration_unemployed_weeks_mean,
            "equilibrium_instrument_rate": instrument_hist[-1],
            "hc_mean": hc_mean,
            "hc_employed_mean": hc_employed_mean,
            "hc_nonemployed_mean": hc_nonemployed_mean,
            "wage_hc_factor_displaced_mean": wage_hc_factor_displaced_mean,
            "wage_hc_factor_nondisplaced_mean": wage_hc_factor_nondisplaced_mean,
            "job_finding_rate_searching_mean": job_finding_rate_searching_mean,
            "job_finding_rate_searching_all_mean": job_finding_rate_searching_all_mean,
            "job_finding_rate_searching_loss_mean": job_finding_rate_searching_loss_mean,
            "log_consumption_employed_mean": log_consumption_employed_mean,
            "log_consumption_nonemployed_mean": log_consumption_nonemployed_mean,
            "marginal_utility_nonemployed_mean": marginal_utility_nonemployed_mean,
            "net_government_spending_working": net_government_spending_working,
            "net_government_spending_all": net_government_spending_all,
            "policy_consumption_employed": policy_consumption_employed,
            "policy_consumption_unemployed": policy_consumption_unemployed,
            "policy_consumption_unemployed_loss": policy_consumption_unemployed_loss,
            "policy_effort_searching": policy_effort_searching,
            "policy_effort_searching_loss": policy_effort_searching_loss,
            "pv_government_spending": pv_government_spending,
            "share_employed": share_employed,
            "share_nonemployed": share_nonemployed,
            "share_searching": share_searching,
            "share_unemployed": share_unemployed,
            "share_unemployed_loss": share_unemployed_loss,
            "total_benefits": total_benefits,
            "ui_benefits_mean": ui_benefits_mean,
            "ui_effective_replacement_rate_mean": ui_effective_replacement_rate,
            "ui_share_floor_binding": ui_share_floor_binding,
            "ui_share_cap_binding": ui_share_cap_binding,
            "wage_hc_factor_employed_mean": wage_hc_factor_employed_mean,
            "wage_hc_factor_nonemployed_mean": wage_hc_factor_nonemployed_mean,
            "wage_hc_factor_pre_displacement_mean": wage_hc_factor_pre_displacement_mean,
            "wage_loss_median": wage_loss_median,
            "welfare": pv_utility_corrected,
        }

        for item in out:
            try:
                out[item] = out[item].tolist()
            except AttributeError:
                pass

    else:
        out = {
            "average_pv_cost_computed": average_pv_cost_computed,
            "average_pv_utility_computed": average_pv_utility_computed,
            "average_pv_utility_computed_corrected": average_pv_utility_computed_corrected,
            "equilibrium_instrument_rate": instrument_hist[-1],
            "policy_consumption_employed": policy_consumption_employed,
            "policy_consumption_unemployed": policy_consumption_unemployed,
            "policy_consumption_unemployed_loss": policy_consumption_unemployed_loss,
            "policy_effort_searching": policy_effort_searching,
            "policy_effort_searching_loss": policy_effort_searching_loss,
            "welfare": pv_utility_corrected,
        }

    if show_progress:
        print("end main")

    return out


#####################################################
# SCRIPT
#####################################################

if __name__ == "__main__":

    try:
        setup_name = sys.argv[1]
        method = sys.argv[2]
    except IndexError:
        setup_name = "edu_high_flat_no_caps"
        method = "linear"

    # load calibration and set some variables
    calibration = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_solve_max": 20,
        "n_simulations": int(1e4),
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_progress_solve": True,
        "show_summary": True,
        "tolerance_solve": 1e-7,
    }

    # solve and simulate
    results = _solve_and_simulate(controls, calibration)

    # store results
    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_" + setup_name + "_results_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)
