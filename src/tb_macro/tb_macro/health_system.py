from typing import Tuple
from jax import numpy as jnp
import pandas as pd
import numpy as np

from summer3.epi import (
    TransitionFlow,
    Stratification,
    CompartmentalEpiModel,
)
from summer3.graph import defer, Time, Parameter

from tb_macro.utils import tanh_based_scaleup, get_scale_data, get_cos_multicurve
from tb_macro.demography import make_multi_interp_array_func


def add_detection(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    clin_strat: Stratification,
    start_time: float,
):
    """Add the process of disease detection to the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        clin_strat: The clinical stratification object
        start_time: The model starting time as a calendar year
    """

    def detect_curve(t, detect_time_1, detect_val_1):
        times = get_scale_data(np.array([1950.0, detect_time_1]))
        vals = get_scale_data(np.array([0.0, detect_val_1]))
        return get_cos_multicurve(t, times, vals)

    peak_detect_time = Parameter("peak_detect_time", 0.0)
    peak_detect_val = Parameter("peak_detect_val", 0.0)
    sim_time = Time + start_time
    detect_func = defer(detect_curve)(sim_time, peak_detect_time, peak_detect_val)

    source = (disease_state["active"], clin_strat["clin"])
    dest = disease_state["treatment"]
    detect = TransitionFlow("detection", source, dest, detect_func)
    epi_model.add_flow(detect)


def compute_outcome_props(
    rx_duration: float,
    prop_neg_rx_death: float,
    tsr: float,
    death_rate: np.array,
) -> dict[str, np.array]:
    """Get the numeric values for all the treatment outcomes.

    Args:
        rx_duration: Treatment duration in model time units (years)
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes resulting in death
        tsr: Treatment success rate
        death_rate: Natural death rate

    Returns:
        Treatment outcome proportions for each of the three outcomes
    """
    prop_nat_death_on_rx = 1.0 - jnp.exp(-rx_duration * death_rate)
    req_prop_death_on_rx = (1.0 - tsr) * prop_neg_rx_death
    prop_death_from_rx = jnp.maximum(req_prop_death_on_rx - prop_nat_death_on_rx, 0.0)
    prop_total_death = prop_death_from_rx + prop_nat_death_on_rx
    relapse_prop = jnp.maximum(1.0 - tsr - prop_total_death, 0.0)
    success = jnp.maximum(1.0 - relapse_prop - prop_total_death, 0.0)
    return {"success": success, "relapse": relapse_prop, "rx_death": prop_death_from_rx}


def get_outcome_rate(
    dur: float,
    prop_neg_rx_death: float,
    tsr: float,
    death_rate: np.array,
    age_strat,
) -> np.array:
    """Get the flow rate for a specific treatment outcome.

    Args:
        outcome: The outcome identifier (success, relapse or rx_death)
        dur: Treatment duration in model time units (years)
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes resulting in death
        tsr: Treatment success rate
        death_rate: Natural death rate

    Returns:
        The flow rate for the outcome requested
    """
    outcome_props = compute_outcome_props(dur, prop_neg_rx_death, tsr, death_rate)
    return {
        outcome: age_strat.categories().wrap(result / dur)
        for outcome, result in outcome_props.items()
    }


def add_treatment_flows(
    death_rates: pd.DataFrame,
    start_time: float,
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    clin_strat: Stratification,
    tsr_data: pd.DataFrame,
    death_in_unsucc_data: pd.DataFrame,
):
    """Add treatment-related outcome flows to epi model.

    Args:
        death_rates: The death rate data
        start_time: The model starting time as a calendar year
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification
        clin_strat: The clinical stratification
    """

    # TSR calculations
    tsr_times = get_scale_data(np.array(tsr_data.index))
    tsr_vals = get_scale_data(np.array(tsr_data))
    tsr_func = defer(lambda t: get_cos_multicurve(t, tsr_times, tsr_vals))(Time)

    # Death in unsuccessful outcomes calculations
    death_unsucc_times = get_scale_data(np.array(death_in_unsucc_data.index))
    death_unsucc_vals = get_scale_data(np.array(death_in_unsucc_data))

    def death_unsucc_curve(t):
        return get_cos_multicurve(t, death_unsucc_times, death_unsucc_vals)

    death_unsucc_func = defer(death_unsucc_curve)(Time)

    # Natural death calculations
    death_vals = np.array(death_rates.index)
    death_times = np.array(death_rates)
    death_array_func = make_multi_interp_array_func(death_vals, death_times, start_time)
    death_func = defer(death_array_func)(Time)

    # Other common variables
    all_age_strata = age_strat[age_strat.strata]
    source = (disease_state["treatment"], all_age_strata)

    # Get all outcome rates
    dur = Parameter("rx_duration", 0.0)
    outcome_rates = defer(get_outcome_rate)(dur, death_unsucc_func, tsr_func, death_func, age_strat)

    # Success
    dest = (disease_state["recovered"], all_age_strata)
    flow = TransitionFlow("success", source, dest, outcome_rates["success"])
    epi_model.add_flow(flow)

    # Relapse
    dest = (clin_strat["subclin"], infect_strat["low"], all_age_strata)
    flow = TransitionFlow("relapse", source, dest, outcome_rates["relapse"])
    epi_model.add_flow(flow)

    # Death on treatment
    dest = (disease_state["mtb_naive"], age_strat["0"])
    flow = TransitionFlow("rx_death", source, dest, outcome_rates["rx_death"])
    epi_model.add_flow(flow)
