""" Compute first best solution.

This module computes the first best solution.

"""
#####################################################
# IMPORTS
#####################################################
import json
import sys
import warnings

import numba as nb
import numpy as np
from scipy import interpolate

from bld.project_paths import project_paths_join as ppj

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


def _get_pv_utils(
    pv_utils_employed_next,
    pv_utils_searching_next,
    pv_utils_searching_loss_next,
    wage_hc_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):

    pv_utils_employed_now = (
        wage_level
        * np.repeat(wage_hc_factor_vector[:, :-1], consumption_grid_size).reshape(
            (n_types, hc_grid_size - 1, consumption_grid_size)
        )
        * np.tile(
            marginal_consumption_utility(consumption_grid),
            (hc_grid_size - 1) * n_types,
        ).reshape((n_types, hc_grid_size - 1, consumption_grid_size))
        + discount_factor
        * np.einsum(
            "i, ijk -> ijk",
            (1 - separation_rate_vector[:, period_idx]),
            pv_utils_employed_next[:, 1:, :],
        )
        + discount_factor
        * np.einsum(
            "i, ijk -> ijk",
            separation_rate_vector[:, period_idx],
            pv_utils_searching_next[:, 1:, :],
        )
    )
    pv_utils_unemployed_now = discount_factor * np.einsum(
        "i, ijk -> ijk", (1 - hc_loss_probability), pv_utils_searching_next[:, :-1, :]
    ) + discount_factor * np.einsum(
        "i, ijk -> ijk", hc_loss_probability, pv_utils_searching_loss_next[:, :-1, :]
    )
    pv_utils_unemployed_loss_now = (
        discount_factor * pv_utils_searching_loss_next[:, :-1, :]
    )

    if period_idx > 0:
        pv_utils_employed_loss_now = _interpolate_hc_loss(
            pv_utils_employed_now,
            wage_loss_reference_vector,
            period_idx,
        )
    else:
        pv_utils_employed_loss_now = pv_utils_employed_now

    # original code
    # (
    #     policy_effort_searching_now,
    #     policy_effort_searching_loss_now,
    #     pv_utils_searching_now,
    #     pv_utils_searching_loss_now
    # ) = _get_search_effort_policy(
    #     pv_utils_employed_now,
    #     pv_utils_employed_loss_now,
    #     pv_utils_unemployed_now,
    #     pv_utils_unemployed_loss_now,
    # )

    # numba
    (policy_effort_searching_now, pv_utils_searching_now) = _solve_searching_iter(
        pv_utils_employed_now,
        pv_utils_unemployed_now,
    )
    (
        policy_effort_searching_loss_now,
        pv_utils_searching_loss_now,
    ) = _solve_searching_iter(pv_utils_employed_loss_now, pv_utils_unemployed_loss_now)

    # FOC
    # (policy_effort_searching_now, pv_utils_searching_now) = _solve_searching_foc(
    #     pv_utils_employed_now, pv_utils_unemployed_now,
    # )
    # (policy_effort_searching_loss_now, pv_utils_searching_loss_now) = _solve_searching_foc(
    #     pv_utils_employed_loss_now, pv_utils_unemployed_loss_now
    # )

    return (
        pv_utils_employed_now,
        pv_utils_searching_now,
        pv_utils_searching_loss_now,
        policy_effort_searching_now,
        policy_effort_searching_loss_now,
    )


def _get_search_effort_policy(
    pv_utils_employed_now,
    pv_utils_employed_loss_now,
    pv_utils_unemployed_now,
    pv_utils_unemployed_loss_now,
):

    pv_utils_searching_now = np.full(
        (n_types, hc_grid_size - 1, consumption_grid_size), np.nan
    )
    pv_utils_searching_loss_now = np.full(
        (n_types, hc_grid_size - 1, consumption_grid_size), np.nan
    )
    policy_effort_searching_now = np.full(
        (n_types, hc_grid_size - 1, consumption_grid_size), np.nan
    )
    policy_effort_searching_loss_now = np.full(
        (n_types, hc_grid_size - 1, consumption_grid_size), np.nan
    )

    for type_idx in range(n_types):
        for consumption_level in range(consumption_grid_size):
            # value function of searching
            returns_searching = (
                np.tile(leisure_utility_on_search_grid, hc_grid_size - 1).reshape(
                    hc_grid_size - 1, search_effort_grid_size
                )
                + np.outer(
                    pv_utils_employed_now[type_idx, :, consumption_level],
                    job_finding_probability_grid,
                )
                + np.outer(
                    pv_utils_unemployed_now[type_idx, :, consumption_level],
                    (1 - job_finding_probability_grid),
                )
            )
            search_effort_idx = returns_searching.argmax(axis=1)
            pv_utils_searching_now[type_idx, :, consumption_level] = np.array(
                [
                    returns_searching[row, col]
                    for row, col in enumerate(search_effort_idx)
                ]
            )
            policy_effort_searching_now[
                type_idx, :, consumption_level
            ] = search_effort_grid[search_effort_idx].T

            # value function of searching low HK unemployed
            returns_searching_loss = (
                np.tile(leisure_utility_on_search_grid, hc_grid_size - 1).reshape(
                    hc_grid_size - 1, search_effort_grid_size
                )
                + np.outer(
                    pv_utils_employed_loss_now[type_idx, :, consumption_level],
                    job_finding_probability_grid,
                )
                + np.outer(
                    pv_utils_unemployed_loss_now[type_idx, :, consumption_level],
                    (1 - job_finding_probability_grid),
                )
            )
            search_effort_idx = returns_searching_loss.argmax(axis=1)
            pv_utils_searching_loss_now[type_idx, :, consumption_level] = np.array(
                [
                    returns_searching_loss[row, col]
                    for row, col in enumerate(search_effort_idx)
                ]
            )
            policy_effort_searching_loss_now[
                type_idx, :, consumption_level
            ] = search_effort_grid[search_effort_idx].T

    return (
        policy_effort_searching_now,
        policy_effort_searching_loss_now,
        pv_utils_searching_now,
        pv_utils_searching_loss_now,
    )


def _get_pv_income(
    pv_income_employed_next,
    pv_income_searching_next,
    pv_income_searching_loss_next,
    policy_effort_searching_now,
    policy_effort_searching_loss_now,
    wage_hc_factor_vector,
    wage_loss_reference_vector,
    period_idx,
):
    pv_income_employed_now = (
        wage_level
        * np.repeat(wage_hc_factor_vector[:, :-1], consumption_grid_size).reshape(
            (n_types, hc_grid_size - 1, consumption_grid_size)
        )
        + discount_factor
        * np.einsum(
            "i, ijk -> ijk",
            (1 - separation_rate_vector[:, period_idx]),
            pv_income_employed_next[:, 1:, :],
        )
        + discount_factor
        * np.einsum(
            "i, ijk -> ijk",
            separation_rate_vector[:, period_idx],
            pv_income_searching_next[:, 1:, :],
        )
    )
    pv_income_unemployed_now = discount_factor * np.einsum(
        "i, ijk -> ijk", (1 - hc_loss_probability), pv_income_searching_next[:, :-1, :]
    ) + discount_factor * np.einsum(
        "i, ijk -> ijk", hc_loss_probability, pv_income_searching_loss_next[:, :-1, :]
    )
    pv_income_unemployed_loss_now = (
        discount_factor * pv_income_searching_loss_next[:, :-1, :]
    )
    if period_idx > 0:
        pv_income_employed_loss_now = _interpolate_hc_loss(
            pv_income_employed_now,
            wage_loss_reference_vector,
            period_idx,
        )
    else:
        pv_income_employed_loss_now = pv_income_employed_now

    pv_income_searching_now = (
        policy_effort_searching_now * pv_income_employed_now
        + (1 - policy_effort_searching_now) * pv_income_unemployed_now
    )
    pv_income_searching_loss_now = (
        policy_effort_searching_loss_now * pv_income_employed_loss_now
        + (1 - policy_effort_searching_loss_now) * pv_income_unemployed_loss_now
    )

    return (
        pv_income_employed_now,
        pv_income_searching_now,
        pv_income_searching_loss_now,
    )


def _get_pv_search_cost(
    pv_search_cost_employed_next,
    pv_search_cost_searching_next,
    pv_search_cost_searching_loss_next,
    policy_effort_searching_now,
    policy_effort_searching_loss_now,
    wage_loss_reference_vector,
    period_idx,
):
    pv_search_cost_employed_now = discount_factor * np.einsum(
        "i, ijk -> ijk",
        (1 - separation_rate_vector[:, period_idx]),
        pv_search_cost_employed_next[:, 1:, :],
    ) + discount_factor * np.einsum(
        "i, ijk -> ijk",
        separation_rate_vector[:, period_idx],
        pv_search_cost_searching_next[:, 1:, :],
    )
    pv_search_cost_unemployed_now = discount_factor * np.einsum(
        "i, ijk -> ijk",
        (1 - hc_loss_probability),
        pv_search_cost_searching_next[:, :-1, :],
    ) + discount_factor * np.einsum(
        "i, ijk -> ijk",
        hc_loss_probability,
        pv_search_cost_searching_loss_next[:, :-1, :],
    )
    pv_search_cost_unemployed_loss_now = (
        discount_factor * pv_search_cost_searching_loss_next[:, :-1, :]
    )
    if period_idx > 0:
        pv_search_cost_employed_loss_now = _interpolate_hc_loss(
            pv_search_cost_employed_now,
            wage_loss_reference_vector,
            period_idx,
        )
    else:
        pv_search_cost_employed_loss_now = pv_search_cost_employed_now

    pv_search_cost_searching_now = (
        leisure_utility_interpolated(policy_effort_searching_now)
        + policy_effort_searching_now * pv_search_cost_employed_now
        + (1 - policy_effort_searching_now) * pv_search_cost_unemployed_now
    )
    pv_search_cost_searching_loss_now = (
        leisure_utility_interpolated(policy_effort_searching_loss_now)
        + policy_effort_searching_loss_now * pv_search_cost_employed_loss_now
        + (1 - policy_effort_searching_loss_now) * pv_search_cost_unemployed_loss_now
    )

    return (
        pv_search_cost_employed_now,
        pv_search_cost_searching_now,
        pv_search_cost_searching_loss_now,
    )


def _interpolate_hc_loss(
    array_in,
    wage_loss_reference_vector,
    period_idx,
):

    # initiate output array
    array_out = np.full(array_in.shape, np.nan)

    # get human capital after depreciation
    hc_after_loss = _hc_after_loss_n_agents(
        np.tile(hc_grid[: period_idx + 1], array_in.shape[0]).reshape(
            array_in.shape[0], -1
        ),
        wage_loss_factor_vector,
        wage_loss_reference_vector,
        period_idx,
    )

    # interpolate output array
    for type_idx in range(n_types):
        for consumption_idx in range(consumption_grid_size):
            array_out[
                type_idx, : period_idx + 1, consumption_idx
            ] = interpolate.interp1d(
                hc_grid[: period_idx + 1],
                array_in[type_idx, : period_idx + 1, consumption_idx],
                kind=interpolation_method,
            )(
                hc_after_loss[type_idx, :]
            )

    return array_out


@nb.njit
def _solve_searching_iter(
    value_employed_now,
    value_unemployed_now,
):
    # initiate objects
    policy = np.full((n_types, hc_grid_size - 1, consumption_grid_size), np.nan)
    value = np.full((n_types, hc_grid_size - 1, consumption_grid_size), np.nan)

    # solve for optimal search effort using grid search method
    for type_idx in range(n_types):
        for experience_level in range(hc_grid_size - 1):
            for consumptions_level in range(consumption_grid_size):

                search_returns = (
                    leisure_utility_on_search_grid
                    + job_finding_probability_grid
                    * np.full(
                        search_effort_grid_size,
                        value_employed_now[
                            type_idx, experience_level, consumptions_level
                        ],
                    )
                    + (1 - job_finding_probability_grid)
                    * np.full(
                        search_effort_grid_size,
                        value_unemployed_now[
                            type_idx, experience_level, consumptions_level
                        ],
                    )
                )
                search_effort_idx = search_returns.argmax()
                value[type_idx, experience_level, consumptions_level] = search_returns[
                    search_effort_idx
                ]
                policy[
                    type_idx, experience_level, consumptions_level
                ] = search_effort_grid[search_effort_idx]

    return policy, value


@nb.njit
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
            for z_idx in range(dims[2]):
                array_out[x_idx, y_idx, z_idx] = search_effort_grid[
                    np.abs(array_in[x_idx, y_idx, z_idx] - search_effort_grid).argmin()
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


def wage_hc_factor_interpolated_1_agent(x, wage_hc_factor_vector):
    return interpolate.interp1d(
        hc_grid,
        wage_hc_factor_vector,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )(x)


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
def marginal_consumption_utility(x):
    return x ** (-risk_aversion_coefficient)


def wage_hc_factor_interpolated(hc_in, wage_hc_factor_vector):

    wage_hc_factor_out = np.full(hc_in.shape, np.nan)

    # iterate over types and interpolate wage_hc_factor_vector
    for idx in range(hc_in.shape[0]):
        wage_hc_factor_out[idx, :] = interpolate.interp1d(
            hc_grid,
            wage_hc_factor_vector[idx, :],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(hc_in[idx, ...])

    return wage_hc_factor_out


def effort_searching_interpolated(
    hc_in, period_idx, policy_effort_searching, consumption_opt_idx
):

    effort_out = np.full(hc_in.shape, np.nan)

    for type_idx in range(hc_in.shape[0]):
        interpolator_lower = interpolate.interp1d(
            hc_grid[:-1],
            policy_effort_searching[
                type_idx, :-1, int(consumption_opt_idx - 1), period_idx
            ],
            kind=interpolation_method,
        )
        interpolator_upper = interpolate.interp1d(
            hc_grid[:-1],
            policy_effort_searching[
                type_idx, :-1, int(consumption_opt_idx), period_idx
            ],
            kind=interpolation_method,
        )

        effort_out[type_idx, :] = (1 - interpolation_weight) * interpolator_lower(
            hc_in[type_idx, :]
        ) + interpolation_weight * interpolator_upper(hc_in[type_idx, :])

    return effort_out


def _solve_first_best(calibration, controls):

    global contact_rate
    global consumption_grid
    global consumption_grid_size
    global discount_factor
    global hc_grid
    global hc_grid_size
    global hc_max
    global hc_loss_probability
    global interpolation_method
    global interpolation_weight
    global job_finding_probability_grid
    global leisure_grid
    global leisure_utility
    global leisure_utility_dx
    global leisure_utility_dxdx
    global leisure_utility_dx_max
    global leisure_utility_dx_min
    global leisure_utility_on_search_grid
    global n_types
    global risk_aversion_coefficient
    global search_effort_grid
    global search_effort_grid_size
    global separation_rate_vector
    global wage_hc_factor_vector_average
    global wage_level
    global wage_loss_factor_vector

    # load controls
    interpolation_method = controls["interpolation_method"]
    n_simulations = controls["n_simulations"]
    seed_simulations = controls["seed_simulation"]
    show_summary = controls["show_summary"]

    # load calibration
    contact_rate = calibration["contact_rate"]
    discount_factor = calibration["discount_factor"]
    hc_loss_probability = np.array(calibration["hc_loss_probability"])
    leisure_utility = np.array(calibration["leisure_utility"])
    leisure_utility_dx = np.array(calibration["leisure_utility_dx"])
    leisure_utility_dxdx = np.array(calibration["leisure_utility_dxdx"])
    n_periods_retired = calibration["n_periods_retired"]
    n_periods_working = calibration["n_periods_working"]
    n_types = calibration["n_types"]
    risk_aversion_coefficient = calibration["risk_aversion_coefficient"]
    search_effort_grid_size = calibration["search_effort_grid_size"]
    search_effort_max = calibration["search_effort_max"]
    search_effort_min = calibration["search_effort_min"]
    separation_rate_vector = np.array(calibration["separation_rate_vector"])
    type_weights = np.array(calibration["type_weights"])
    wage_hc_factor_vector = np.array(calibration["wage_hc_factor_vector"])
    wage_level = calibration["wage_level"]
    wage_loss_factor_vector = np.array(calibration["wage_loss_factor_vector"])
    wage_loss_reference_vector = np.array(calibration["wage_loss_reference_vector"])

    consumption_grid_size = 51

    # calculate derived parameters
    hc_grid = np.arange(n_periods_working + 1)
    hc_grid_size = len(hc_grid)
    hc_max = np.amax(hc_grid)
    leisure_grid = np.linspace(
        search_effort_min, search_effort_max, len(leisure_utility)
    )
    leisure_utility_dx_max = leisure_utility_dx_interpolated(search_effort_max)
    leisure_utility_dx_min = leisure_utility_dx_interpolated(search_effort_min)
    search_effort_grid = np.linspace(
        search_effort_min, search_effort_max, search_effort_grid_size
    )
    job_finding_probability_grid = job_finding_probability(search_effort_grid)
    leisure_utility_on_search_grid = leisure_utility_interpolated(search_effort_grid)
    wage_hc_factor_vector_average = np.average(
        wage_hc_factor_vector, weights=type_weights, axis=0
    )

    dif0h = np.linspace(0.01, 1, int((consumption_grid_size - 1) / 2)) ** 1.8
    dif0l = -np.linspace(1, 0.01, int((consumption_grid_size - 1) / 2)) ** 1.8

    consumption_grid = np.concatenate(
        (
            -dif0l * 0.7 + (1 + dif0l) * 1.371663,
            np.array([1.371663]),
            (1 - dif0h) * 1.371663 + dif0h * 1.9,
        )
    )  # this vector is more detailed around 1.371663

    # (I) solution

    # initiate objects to store value functions and policy functions
    pv_utils_employed = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_utils_searching = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_utils_searching_loss = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )

    policy_effort_searching = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    policy_effort_searching_loss = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )

    pv_income_employed = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_income_searching = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_income_searching_loss = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )

    pv_search_cost_employed = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_search_cost_searching = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )
    pv_search_cost_searching_loss = np.full(
        (n_types, hc_grid_size, consumption_grid_size, n_periods_working + 1),
        np.nan,
    )

    # initiate first period of retirement
    pv_utils_employed[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_utils_searching[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_utils_searching_loss[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )

    pv_income_employed[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_income_searching[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_income_searching_loss[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )

    pv_search_cost_employed[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_search_cost_searching[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )
    pv_search_cost_searching_loss[:, :, :, -1] = np.zeros(
        (n_types, hc_grid_size, consumption_grid_size)
    )

    # iterate over periods

    period = n_periods_working
    period_idx = period - 1

    while period > 0:

        # load policy and pv functions for the next period
        pv_utils_employed_next = pv_utils_employed[:, :, :, period_idx + 1]
        pv_utils_searching_next = pv_utils_searching[:, :, :, period_idx + 1]
        pv_utils_searching_loss_next = pv_utils_searching_loss[:, :, :, period_idx + 1]

        pv_income_employed_next = pv_income_employed[:, :, :, period_idx + 1]
        pv_income_searching_next = pv_income_searching[:, :, :, period_idx + 1]
        pv_income_searching_loss_next = pv_income_searching_loss[
            :, :, :, period_idx + 1
        ]

        pv_search_cost_employed_next = pv_search_cost_employed[:, :, :, period_idx + 1]
        pv_search_cost_searching_next = pv_search_cost_searching[
            :, :, :, period_idx + 1
        ]
        pv_search_cost_searching_loss_next = pv_search_cost_searching_loss[
            :, :, :, period_idx + 1
        ]

        # present value functions of income in utils
        (
            pv_utils_employed_now,
            pv_utils_searching_now,
            pv_utils_searching_loss_now,
            policy_effort_searching_now,
            policy_effort_searching_loss_now,
        ) = _get_pv_utils(
            pv_utils_employed_next,
            pv_utils_searching_next,
            pv_utils_searching_loss_next,
            wage_hc_factor_vector,
            wage_loss_reference_vector,
            period_idx,
        )

        # present value of income in consumption units functions
        (
            pv_income_employed_now,
            pv_income_searching_now,
            pv_income_searching_loss_now,
        ) = _get_pv_income(
            pv_income_employed_next,
            pv_income_searching_next,
            pv_income_searching_loss_next,
            policy_effort_searching_now,
            policy_effort_searching_loss_now,
            wage_hc_factor_vector,
            wage_loss_reference_vector,
            period_idx,
        )

        # present value of search effort cost in utils functions
        (
            pv_search_cost_employed_now,
            pv_search_cost_searching_now,
            pv_search_cost_searching_loss_now,
        ) = _get_pv_search_cost(
            pv_search_cost_employed_next,
            pv_search_cost_searching_next,
            pv_search_cost_searching_loss_next,
            policy_effort_searching_now,
            policy_effort_searching_loss_now,
            wage_loss_reference_vector,
            period_idx,
        )

        # store results
        pv_utils_employed[:, :-1, :, period_idx] = pv_utils_employed_now
        pv_utils_searching[:, :-1, :, period_idx] = pv_utils_searching_now
        pv_utils_searching_loss[:, :-1, :, period_idx] = pv_utils_searching_loss_now

        policy_effort_searching[:, :-1, :, period_idx] = policy_effort_searching_now
        policy_effort_searching_loss[
            :, :-1, :, period_idx
        ] = policy_effort_searching_loss_now

        pv_income_employed[:, :-1, :, period_idx] = pv_income_employed_now
        pv_income_searching[:, :-1, :, period_idx] = pv_income_searching_now
        pv_income_searching_loss[:, :-1, :, period_idx] = pv_income_searching_loss_now

        pv_search_cost_employed[:, :-1, :, period_idx] = pv_search_cost_employed_now
        pv_search_cost_searching[:, :-1, :, period_idx] = pv_search_cost_searching_now
        pv_search_cost_searching_loss[
            :, :-1, :, period_idx
        ] = pv_search_cost_searching_loss_now

        # initiate next iteration
        period -= 1
        period_idx = period - 1

    # average over types
    # pv_utils_employed_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_utils_employed
    # )
    # pv_utils_searching_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_utils_searching
    # )
    # pv_utils_searching_loss_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_utils_searching_loss
    # )
    #
    # pv_income_employed_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_income_employed
    # )
    pv_income_searching_aggregated = np.einsum(
        "i, ijkl -> jkl", type_weights, pv_income_searching
    )
    # pv_income_searching_loss_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_income_searching_loss
    # )
    #
    # pv_search_cost_employed_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_search_cost_employed
    # )
    pv_search_cost_searching_aggregated = np.einsum(
        "i, ijkl -> jkl", type_weights, pv_search_cost_searching
    )
    # pv_search_cost_searching_loss_aggregated = np.einsum(
    #     "i, ijkl -> jkl", type_weights, pv_search_cost_searching_loss
    # )

    # compute some outcomes

    # Finding the consumption that makes income = consumption
    consumption_opt_first_best = interpolate.interp1d(
        pv_income_searching_aggregated[0, :, 0]
        - consumption_grid
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor),
        consumption_grid,
        kind=interpolation_method,
    )(0)
    consumption_opt_idx = interpolate.interp1d(
        pv_income_searching_aggregated[0, 0, 0]
        - consumption_grid
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor),
        np.linspace(1, consumption_grid_size, consumption_grid_size),
        kind=interpolation_method,
    )(0)
    consumption_opt_idx = np.ceil(consumption_opt_idx)

    pv_consumption_computed = (
        consumption_opt_first_best
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor)
    )
    pv_income_computed = interpolate.interp1d(
        pv_income_searching_aggregated[0, :, 0]
        - consumption_grid
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor),
        pv_income_searching_aggregated[0, :, 0],
        kind=interpolation_method,
    )(0)

    average_pv_utility_computed = consumption_utility(consumption_opt_first_best) * (
        1 - discount_factor ** (n_periods_working + n_periods_retired)
    ) / (1 - discount_factor) + interpolate.interp1d(
        consumption_grid,
        pv_search_cost_searching_aggregated[0, :, 0],
        kind=interpolation_method,
    )(
        consumption_opt_first_best
    )
    pv_utility_computed = np.repeat(average_pv_utility_computed, n_types).reshape(
        (n_types,)
    )

    if show_summary:
        summary_solve = np.array(
            (
                ("optimal consumption level", np.round(consumption_opt_first_best, 5)),
                ("PV consumption (computed)", np.round(pv_consumption_computed, 5)),
                ("PV income (computed)", np.round(pv_income_computed, 5)),
                ("PV utility (computed)", np.round(average_pv_utility_computed, 5)),
            )
        )
        print(summary_solve)

    # (II) simulation

    # initiate simulation study
    np.random.seed(seed_simulations)

    interpolation_weight = (
        consumption_grid[int(consumption_opt_idx - 1)] - consumption_opt_first_best
    ) / (
        consumption_grid[int(consumption_opt_idx - 1)]
        - consumption_grid[int(consumption_opt_idx)]
    )

    # status tracker
    employed_simulated = np.full((n_types, n_simulations), 0.0).astype(bool)
    searching_simulated = np.full((n_types, n_simulations), 1.0).astype(bool)
    searching_loss_simulated = np.full((n_types, n_simulations), 0.0).astype(bool)

    # human capital tracker
    hc_simulated = np.full((n_types, n_simulations), 0.0)
    hc_pre_displacement_simulated = np.full((n_types, n_simulations), 0.0)

    # tracker for present value of income and search cost
    pv_income_simulated = np.full((n_types, n_simulations), 0.0)
    pv_search_cost_simulated = np.full((n_types, n_simulations), 0.0)

    # summary statistics
    effort_searching_all_mean = np.full((n_types, n_periods_working), np.nan)
    effort_searching_mean = np.full((n_types, n_periods_working), np.nan)
    effort_searching_loss_mean = np.full((n_types, n_periods_working), np.nan)
    share_unemployed_mean = np.full((n_types, n_periods_working), np.nan)
    share_unemployed_loss_mean = np.full((n_types, n_periods_working), np.nan)
    wage_employed_mean = np.full((n_types, n_periods_working), np.nan)
    wage_nonemployed_mean = np.full((n_types, n_periods_working), np.nan)
    wage_unemployed_loss_mean = np.full((n_types, n_periods_working), np.nan)
    wage_pre_displacement_nonemployed_mean = np.full(
        (n_types, n_periods_working), np.nan
    )

    # iterate forward
    period = 0

    while period < n_periods_working:

        period += 1
        period_idx = period - 1

        # (i) search phase

        # simulate search effort
        effort_searching_simulated = effort_searching_interpolated(
            hc_simulated,
            period_idx,
            policy_effort_searching,
            consumption_opt_idx,
        )
        effort_searching_loss_simulated = effort_searching_interpolated(
            hc_simulated,
            period_idx,
            policy_effort_searching_loss,
            consumption_opt_idx,
        )

        job_finding_probability_searching_simulated = job_finding_probability(
            effort_searching_simulated
        )
        job_finding_probability_searching_loss_simulated = job_finding_probability(
            effort_searching_loss_simulated
        )

        # compute search phase statistics
        effort_searching_mean[:, period_idx] = conditional_mean(
            effort_searching_simulated, searching_simulated, axis=1
        )
        effort_searching_loss_mean[:, period_idx] = conditional_mean(
            effort_searching_loss_simulated, searching_loss_simulated, axis=1
        )
        effort_searching_all_mean[:, period_idx] = np.average(
            np.array(
                [
                    effort_searching_mean[:, period_idx],
                    effort_searching_loss_mean[:, period_idx],
                ]
            ),
            weights=np.array(
                [
                    np.sum(searching_simulated, axis=1),
                    np.sum(searching_loss_simulated, axis=1),
                ]
            ),
            axis=0,
        )

        # update present value of simulated search cost (in utils)
        pv_search_cost_simulated += discount_factor ** period_idx * (
            searching_simulated
            * leisure_utility_interpolated(
                effort_searching_interpolated(
                    hc_simulated,
                    period_idx,
                    policy_effort_searching,
                    consumption_opt_idx,
                )
            )
            + searching_loss_simulated
            * leisure_utility_interpolated(
                effort_searching_interpolated(
                    hc_simulated,
                    period_idx,
                    policy_effort_searching_loss,
                    consumption_opt_idx,
                )
            )
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

        # simulate hc transition to consumption phase
        hc_simulated = (
            hc_simulated
            - (
                hc_simulated
                - _hc_after_loss_n_agents(
                    hc_simulated,
                    wage_loss_factor_vector,
                    wage_loss_reference_vector,
                    period_idx,
                )
            )
            * searching_loss_simulated
            * job_finding_event_searching_loss_simulated
        )  # hc loss materializes upon reemployment

        # simulate state transition to consumption phase
        employed_simulated = (
            employed_simulated
            + searching_simulated * job_finding_event_searching_simulated
            + searching_loss_simulated * job_finding_event_searching_loss_simulated
        )
        unemployed_simulated = searching_simulated * (
            1 - job_finding_event_searching_simulated
        )
        unemployed_loss_simulated = searching_loss_simulated * (
            1 - job_finding_event_searching_loss_simulated
        )
        nonemployed_simulated = unemployed_simulated + unemployed_loss_simulated

        # check for error in state simulation
        if (
            np.sum(
                unemployed_simulated + unemployed_loss_simulated + employed_simulated
            )
            < n_simulations
        ):
            warnings.warn(
                f"ERROR! in transition from search phase "
                f"to consumption phase in period {period_idx}"
            )

        # (ii) consumption phase

        # compute statistics for consumption phase
        share_unemployed_mean[:, period_idx] = np.mean(unemployed_simulated, axis=1)
        share_unemployed_loss_mean[:, period_idx] = np.mean(
            unemployed_loss_simulated, axis=1
        )

        wage_simulated = wage_level * wage_hc_factor_interpolated(
            hc_simulated, wage_hc_factor_vector
        )
        wage_pre_displacement_simulated = wage_level * wage_hc_factor_interpolated(
            hc_pre_displacement_simulated, wage_hc_factor_vector
        )

        wage_employed_mean[:, period_idx] = conditional_mean(
            wage_simulated, employed_simulated, axis=1
        )
        wage_nonemployed_mean[:, period_idx] = conditional_mean(
            wage_simulated, nonemployed_simulated, axis=1
        )
        wage_unemployed_loss_mean[:, period_idx] = conditional_mean(
            wage_simulated, unemployed_loss_simulated, axis=1
        )
        wage_pre_displacement_nonemployed_mean[:, period_idx] = conditional_mean(
            wage_pre_displacement_simulated, nonemployed_simulated, axis=1
        )

        # update present value of simulated income (in consumption units)
        pv_income_simulated = (
            pv_income_simulated
            + discount_factor ** period_idx
            * wage_hc_factor_interpolated(
                hc_simulated, wage_level * wage_hc_factor_vector
            )
            * employed_simulated
        )

        # generate transition events
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

        # simulate hc transition to next period
        hc_simulated = (
            hc_simulated + np.full((n_types, n_simulations), 1.0) * employed_simulated
        )
        hc_pre_displacement_simulated = (
            hc_pre_displacement_simulated * nonemployed_simulated
            + hc_simulated * employed_simulated
        )

        # simulate state transition to search phase in next period
        searching_loss_simulated = (
            unemployed_loss_simulated + unemployed_simulated * hc_loss_event_simulated
        ).astype(bool)
        searching_simulated = (
            unemployed_simulated * (1 - hc_loss_event_simulated)
            + employed_simulated * job_loss_event_simulated
        ).astype(bool)
        employed_simulated = (
            employed_simulated * (1 - job_loss_event_simulated)
        ).astype(bool)

        # check for error in state simulation
        if (
            np.sum(searching_simulated + searching_loss_simulated + employed_simulated)
            < n_simulations
        ):
            warnings.warn(
                f"ERROR! in transition from consumption phase "
                f"in period {period_idx} to search phase in {period_idx + 1}"
            )

    # average over types
    share_unemployed_aggregated = np.average(
        share_unemployed_mean, weights=type_weights, axis=0
    )
    share_unemployed_loss_aggregated = np.average(
        share_unemployed_loss_mean, weights=type_weights, axis=0
    )

    effort_searching_aggregated = np.average(
        effort_searching_mean, weights=type_weights, axis=0
    )
    effort_searching_loss_aggregated = np.average(
        effort_searching_loss_mean, weights=type_weights, axis=0
    )
    effort_searching_all_aggregated = np.average(
        effort_searching_all_mean, weights=type_weights, axis=0
    )

    # compute some outcomes

    # consumption (as computed)
    pv_consumption_simulated = np.repeat(
        consumption_opt_first_best
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor),
        n_types,
    )
    average_pv_consumption_simulated = np.average(
        pv_consumption_simulated, weights=type_weights, axis=0
    )

    # income
    pv_income_mean = np.mean(pv_income_simulated, axis=1)
    average_pv_income_simulated = np.average(
        pv_income_mean, weights=type_weights, axis=0
    )

    # search cost
    pv_search_cost_mean = np.mean(pv_search_cost_simulated, axis=1)
    average_pv_search_cost_simulated = np.average(
        pv_search_cost_mean, weights=type_weights, axis=0
    )

    # utility
    pv_utility_simulated = (
        np.repeat(
            consumption_utility(consumption_opt_first_best)
            * (1 - discount_factor ** (n_periods_working + n_periods_retired))
            / (1 - discount_factor),
            n_types,
        )
        + pv_search_cost_mean
    )
    average_pv_utility_simulated = (
        consumption_utility(consumption_opt_first_best)
        * (1 - discount_factor ** (n_periods_working + n_periods_retired))
        / (1 - discount_factor)
        + average_pv_search_cost_simulated
    )

    if show_summary:
        summary_simulate = np.array(
            (
                ("number of simulations", n_simulations),
                (
                    "PV consumption (simulated)",
                    np.round(average_pv_consumption_simulated, 5),
                ),
                ("PV income (simulated)", np.round(average_pv_income_simulated, 5)),
                ("PV utility (simulated)", np.round(average_pv_utility_simulated, 5)),
            )
        )
        print(summary_simulate)

    # implied optimal policies
    ui_replacement_rate_vector_first_best = (
        consumption_opt_first_best / wage_pre_displacement_nonemployed_mean
    )
    income_tax_rate_vector_first_best = (
        1 - consumption_opt_first_best / wage_employed_mean
    )

    # store some results
    out = {
        "consumption_grid": consumption_grid,
        "consumption_opt_first_best": consumption_opt_first_best,
        "consumption_opt_first_best_idx": consumption_opt_idx,
        "effort_searching_all_aggregated": effort_searching_all_aggregated,
        "effort_searching_aggregated": effort_searching_aggregated,
        "effort_searching_loss_aggregated": effort_searching_loss_aggregated,
        "effort_searching_all_mean": effort_searching_all_mean,
        "effort_searching_mean": effort_searching_mean,
        "effort_searching_loss_mean": effort_searching_loss_mean,
        "wage_loss_factor_vector": wage_loss_factor_vector,
        "income_tax_rate_vector_first_best": income_tax_rate_vector_first_best,
        "interpolation_weight": interpolation_weight,
        "n_simulations": n_simulations,
        "share_unemployed": share_unemployed_mean,
        "share_unemployed_loss": share_unemployed_loss_mean,
        "share_unemployed_aggregated": share_unemployed_aggregated,
        "share_unemployed_loss_aggregated": share_unemployed_loss_aggregated,
        "share_unemployed_mean": share_unemployed_mean,
        "share_unemployed_loss_mean": share_unemployed_loss_mean,
        "ui_replacement_rate_vector_first_best": ui_replacement_rate_vector_first_best,
        "wage_employed_mean": wage_employed_mean,
        "wage_pre_displacement_nonemployed_mean": wage_pre_displacement_nonemployed_mean,
        "wage_unemployed_loss_mean": wage_unemployed_loss_mean,
        "wage_hc_factor_vector": wage_hc_factor_vector,
        "wealth": pv_income_computed,
        "wealth_simulated": pv_income_mean,
        "welfare": pv_utility_computed,
        "welfare_simulated": pv_utility_simulated,
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

    # load calibration and set variables
    calibration = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_" + setup_name + ".json"))
    )

    # set controls
    controls = {
        "interpolation_method": method,
        "n_iterations_solve_max": 20,
        "n_simulations": int(1e6),
        "run_simulation": True,
        "seed_simulation": 3405,
        "show_summary": True,
    }

    results = _solve_first_best(calibration, controls)

    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_" + setup_name + "_first_best_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)
