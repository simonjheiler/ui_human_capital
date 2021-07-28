""" Compute first best solution.

This module computes the first best solution.

"""
#####################################################
# IMPORTS
#####################################################
import json
import sys

import numpy as np
from bld.project_paths import project_paths_join as ppj

#####################################################
# PARAMETERS
#####################################################


#####################################################
# FUNCTIONS
#####################################################


def _get_results_3_agents_individual():

    calibration_base = json.load(
        open(ppj("IN_MODEL_SPECS", "analytics_calibration_base_individual.json"))
    )

    first_best_high = json.load(
        open(
            ppj(
                "OUT_RESULTS",
                "analytics",
                "analytics_edu_high_opt_rate_only_first_best_linear.json",
            )
        )
    )
    first_best_medium = json.load(
        open(
            ppj(
                "OUT_RESULTS",
                "analytics",
                "analytics_edu_medium_opt_rate_only_first_best_linear.json",
            )
        )
    )
    first_best_low = json.load(
        open(
            ppj(
                "OUT_RESULTS",
                "analytics",
                "analytics_edu_low_opt_rate_only_first_best_linear.json",
            )
        )
    )

    # load parameters from base calibration
    type_weights = np.array(calibration_base["type_weights"])

    n_simulations = first_best_high["n_simulations"]
    consumption_grid = np.array(first_best_high["consumption_grid"])

    consumption_opt_first_best = np.array(
        [
            first_best_high["consumption_opt_first_best"],
            first_best_medium["consumption_opt_first_best"],
            first_best_low["consumption_opt_first_best"],
        ]
    )
    consumption_opt_first_best_idx = np.array(
        [
            first_best_high["consumption_opt_first_best_idx"],
            first_best_medium["consumption_opt_first_best_idx"],
            first_best_low["consumption_opt_first_best_idx"],
        ]
    )
    effort_searching_all_aggregated = np.array(
        [
            first_best_high["effort_searching_all_aggregated"],
            first_best_medium["effort_searching_all_aggregated"],
            first_best_low["effort_searching_all_aggregated"],
        ]
    )
    effort_searching_aggregated = np.array(
        [
            first_best_high["effort_searching_aggregated"],
            first_best_medium["effort_searching_aggregated"],
            first_best_low["effort_searching_aggregated"],
        ]
    )
    effort_searching_loss_aggregated = np.array(
        [
            first_best_high["effort_searching_loss_aggregated"],
            first_best_medium["effort_searching_loss_aggregated"],
            first_best_low["effort_searching_loss_aggregated"],
        ]
    )
    effort_searching_all_mean = np.array(
        [
            first_best_high["effort_searching_all_mean"],
            first_best_medium["effort_searching_all_mean"],
            first_best_low["effort_searching_all_mean"],
        ]
    )
    effort_searching_mean = np.array(
        [
            first_best_high["effort_searching_mean"],
            first_best_medium["effort_searching_mean"],
            first_best_low["effort_searching_mean"],
        ]
    )
    effort_searching_loss_mean = np.array(
        [
            first_best_high["effort_searching_loss_mean"],
            first_best_medium["effort_searching_loss_mean"],
            first_best_low["effort_searching_loss_mean"],
        ]
    )
    wage_loss_factor_vector = np.array(
        [
            first_best_high["wage_loss_factor_vector"],
            first_best_medium["wage_loss_factor_vector"],
            first_best_low["wage_loss_factor_vector"],
        ]
    )
    income_tax_rate_vector_first_best = np.array(
        [
            first_best_high["income_tax_rate_vector_first_best"],
            first_best_medium["income_tax_rate_vector_first_best"],
            first_best_low["income_tax_rate_vector_first_best"],
        ]
    )
    interpolation_weight = np.array(
        [
            first_best_high["interpolation_weight"],
            first_best_medium["interpolation_weight"],
            first_best_low["interpolation_weight"],
        ]
    )
    share_unemployed_mean = np.array(
        [
            first_best_high["share_unemployed_mean"],
            first_best_medium["share_unemployed_mean"],
            first_best_low["share_unemployed_mean"],
        ]
    )
    share_unemployed_loss_mean = np.array(
        [
            first_best_high["share_unemployed_loss_mean"],
            first_best_medium["share_unemployed_loss_mean"],
            first_best_low["share_unemployed_loss_mean"],
        ]
    )
    ui_replacement_rate_vector_first_best = np.array(
        [
            first_best_high["ui_replacement_rate_vector_first_best"],
            first_best_medium["ui_replacement_rate_vector_first_best"],
            first_best_low["ui_replacement_rate_vector_first_best"],
        ]
    )
    wage_employed_mean = np.array(
        [
            first_best_high["wage_employed_mean"],
            first_best_medium["wage_employed_mean"],
            first_best_low["wage_employed_mean"],
        ]
    )
    wage_pre_displacement_nonemployed_mean = np.array(
        [
            first_best_high["wage_pre_displacement_nonemployed_mean"],
            first_best_medium["wage_pre_displacement_nonemployed_mean"],
            first_best_low["wage_pre_displacement_nonemployed_mean"],
        ]
    )
    wage_unemployed_loss_mean = np.array(
        [
            first_best_high["wage_unemployed_loss_mean"],
            first_best_medium["wage_unemployed_loss_mean"],
            first_best_low["wage_unemployed_loss_mean"],
        ]
    )
    wage_hc_factor_vector = np.array(
        [
            first_best_high["wage_hc_factor_vector"],
            first_best_medium["wage_hc_factor_vector"],
            first_best_low["wage_hc_factor_vector"],
        ]
    )
    wealth = np.array(
        [
            first_best_high["wealth"],
            first_best_medium["wealth"],
            first_best_low["wealth"],
        ]
    )
    wealth_simulated = np.array(
        [
            first_best_high["wealth_simulated"],
            first_best_medium["wealth_simulated"],
            first_best_low["wealth_simulated"],
        ]
    )
    welfare = np.array(
        [
            first_best_high["welfare"],
            first_best_medium["welfare"],
            first_best_low["welfare"],
        ]
    )
    welfare_simulated = np.array(
        [
            first_best_high["welfare_simulated"],
            first_best_medium["welfare_simulated"],
            first_best_low["welfare_simulated"],
        ]
    )

    pv_income_employed = np.array(
        [
            first_best_high["pv_income_employed"],
            first_best_medium["pv_income_employed"],
            first_best_low["pv_income_employed"],
        ]
    )

    pv_income_searching = np.array(
        [
            first_best_high["pv_income_searching"],
            first_best_medium["pv_income_searching"],
            first_best_low["pv_income_searching"],
        ]
    )
    pv_income_searching_loss = np.array(
        [
            first_best_high["pv_income_searching_loss"],
            first_best_medium["pv_income_searching_loss"],
            first_best_low["pv_income_searching_loss"],
        ]
    )
    pv_search_cost_employed = np.array(
        [
            first_best_high["pv_search_cost_employed"],
            first_best_medium["pv_search_cost_employed"],
            first_best_low["pv_search_cost_employed"],
        ]
    )
    pv_search_cost_searching = np.array(
        [
            first_best_high["pv_search_cost_searching"],
            first_best_medium["pv_search_cost_searching"],
            first_best_low["pv_search_cost_searching"],
        ]
    )
    pv_search_cost_searching_loss = np.array(
        [
            first_best_high["pv_search_cost_searching_loss"],
            first_best_medium["pv_search_cost_searching_loss"],
            first_best_low["pv_search_cost_searching_loss"],
        ]
    )
    pv_utils_employed = np.array(
        [
            first_best_high["pv_utils_employed"],
            first_best_medium["pv_utils_employed"],
            first_best_low["pv_utils_employed"],
        ]
    )
    pv_utils_searching = np.array(
        [
            first_best_high["pv_utils_searching"],
            first_best_medium["pv_utils_searching"],
            first_best_low["pv_utils_searching"],
        ]
    )
    pv_utils_searching_loss = np.array(
        [
            first_best_high["pv_utils_searching_loss"],
            first_best_medium["pv_utils_searching_loss"],
            first_best_low["pv_utils_searching_loss"],
        ]
    )
    share_unemployed = np.array(
        [
            first_best_high["share_unemployed"],
            first_best_medium["share_unemployed"],
            first_best_low["share_unemployed"],
        ]
    )
    share_unemployed_loss = np.array(
        [
            first_best_high["share_unemployed_loss"],
            first_best_medium["share_unemployed_loss"],
            first_best_low["share_unemployed_loss"],
        ]
    )

    consumption_opt_first_best = np.squeeze(consumption_opt_first_best)
    consumption_opt_first_best_idx = np.squeeze(consumption_opt_first_best_idx)
    effort_searching_all_aggregated = np.squeeze(effort_searching_all_aggregated)
    effort_searching_aggregated = np.squeeze(effort_searching_aggregated)
    effort_searching_loss_aggregated = np.squeeze(effort_searching_loss_aggregated)
    effort_searching_all_mean = np.squeeze(effort_searching_all_mean)
    effort_searching_mean = np.squeeze(effort_searching_mean)
    effort_searching_loss_mean = np.squeeze(effort_searching_loss_mean)
    wage_loss_factor_vector = np.squeeze(wage_loss_factor_vector)
    income_tax_rate_vector_first_best = np.squeeze(income_tax_rate_vector_first_best)
    interpolation_weight = np.squeeze(interpolation_weight)
    share_unemployed_mean = np.squeeze(share_unemployed_mean)
    share_unemployed_loss_mean = np.squeeze(share_unemployed_loss_mean)
    ui_replacement_rate_vector_first_best = np.squeeze(
        ui_replacement_rate_vector_first_best
    )
    wage_employed_mean = np.squeeze(wage_employed_mean)
    wage_pre_displacement_nonemployed_mean = np.squeeze(
        wage_pre_displacement_nonemployed_mean
    )
    wage_unemployed_loss_mean = np.squeeze(wage_unemployed_loss_mean)
    wage_hc_factor_vector = np.squeeze(wage_hc_factor_vector)
    wealth = np.squeeze(wealth)
    wealth_simulated = np.squeeze(wealth_simulated)
    welfare = np.squeeze(welfare)
    welfare_simulated = np.squeeze(welfare_simulated)
    pv_income_employed = np.squeeze(pv_income_employed)
    pv_income_searching = np.squeeze(pv_income_searching)
    pv_income_searching_loss = np.squeeze(pv_income_searching_loss)
    pv_search_cost_employed = np.squeeze(pv_search_cost_employed)
    pv_search_cost_searching = np.squeeze(pv_search_cost_searching)
    pv_search_cost_searching_loss = np.squeeze(pv_search_cost_searching_loss)
    pv_utils_employed = np.squeeze(pv_utils_employed)
    pv_utils_searching = np.squeeze(pv_utils_searching)
    pv_utils_searching_loss = np.squeeze(pv_utils_searching_loss)
    share_unemployed = np.squeeze(share_unemployed)
    share_unemployed_loss = np.squeeze(share_unemployed_loss)

    pv_income_employed_aggregated = np.average(
        pv_income_employed, weights=type_weights, axis=0
    )
    pv_income_searching_aggregated = np.average(
        pv_income_searching, weights=type_weights, axis=0
    )
    pv_income_searching_loss_aggregated = np.average(
        pv_income_searching_loss, weights=type_weights, axis=0
    )
    pv_search_cost_employed_aggregated = np.average(
        pv_search_cost_employed, weights=type_weights, axis=0
    )
    pv_search_cost_searching_aggregated = np.average(
        pv_search_cost_searching, weights=type_weights, axis=0
    )
    pv_search_cost_searching_loss_aggregated = np.average(
        pv_search_cost_searching_loss, weights=type_weights, axis=0
    )
    pv_utils_employed_aggregated = np.average(
        pv_utils_employed, weights=type_weights, axis=0
    )
    pv_utils_searching_aggregated = np.average(
        pv_utils_searching, weights=type_weights, axis=0
    )
    pv_utils_searching_loss_aggregated = np.average(
        pv_utils_searching_loss, weights=type_weights, axis=0
    )
    share_unemployed_aggregated = np.average(
        share_unemployed, weights=type_weights, axis=0
    )
    share_unemployed_loss_aggregated = np.average(
        share_unemployed_loss, weights=type_weights, axis=0
    )

    out = {
        "consumption_grid": consumption_grid,
        "consumption_opt_first_best": consumption_opt_first_best,
        "consumption_opt_first_best_idx": consumption_opt_first_best_idx,
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
        "pv_income_employed": pv_income_employed,
        "pv_income_searching": pv_income_searching,
        "pv_income_searching_loss": pv_income_searching_loss,
        "pv_search_cost_employed": pv_search_cost_employed,
        "pv_search_cost_searching": pv_search_cost_searching,
        "pv_search_cost_searching_loss": pv_search_cost_searching_loss,
        "pv_utils_employed": pv_utils_employed,
        "pv_utils_searching": pv_utils_searching,
        "pv_utils_searching_loss": pv_utils_searching_loss,
        "share_unemployed": share_unemployed,
        "share_unemployed_loss": share_unemployed_loss,
        "pv_income_employed_aggregated": pv_income_employed_aggregated,
        "pv_income_searching_aggregated": pv_income_searching_aggregated,
        "pv_income_searching_loss_aggregated": pv_income_searching_loss_aggregated,
        "pv_search_cost_employed_aggregated": pv_search_cost_employed_aggregated,
        "pv_search_cost_searching_aggregated": pv_search_cost_searching_aggregated,
        "pv_search_cost_searching_loss_aggregated": pv_search_cost_searching_loss_aggregated,
        "pv_utils_employed_aggregated": pv_utils_employed_aggregated,
        "pv_utils_searching_aggregated": pv_utils_searching_aggregated,
        "pv_utils_searching_loss_aggregated": pv_utils_searching_loss_aggregated,
        "share_unemployed_aggregated": share_unemployed_aggregated,
        "share_unemployed_loss_aggregated": share_unemployed_loss_aggregated,
        "share_unemployed_mean": share_unemployed_mean,
        "share_unemployed_loss_mean": share_unemployed_loss_mean,
        "ui_replacement_rate_vector_first_best": ui_replacement_rate_vector_first_best,
        "wage_employed_mean": wage_employed_mean,
        "wage_pre_displacement_nonemployed_mean": wage_pre_displacement_nonemployed_mean,
        "wage_unemployed_loss_mean": wage_unemployed_loss_mean,
        "wage_hc_factor_vector": wage_hc_factor_vector,
        "wealth": wealth,
        "wealth_simulated": wealth_simulated,
        "welfare": welfare,
        "welfare_simulated": welfare_simulated,
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
        method = sys.argv[2]
    except IndexError:
        method = "linear"

    results = _get_results_3_agents_individual()

    with open(
        ppj(
            "OUT_RESULTS",
            "analytics",
            "analytics_base_individual_first_best_" + method + ".json",
        ),
        "w",
    ) as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)
