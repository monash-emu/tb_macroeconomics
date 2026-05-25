from typing import Tuple
from jax import numpy as jnp
import pandas as pd
import numpy as np

from summer3.epi import (
    TransitionFlow,
    CompartmentalModelODE,
    Stratification,
    CompartmentalEpiModel,
)
from summer3.graph import defer, Time, Parameter

from tb_macro.constants import AGE_STRATA
from tb_macro.utils import tanh_based_scaleup, get_scale_data, get_cos_multicurve
from tb_macro.demography import make_multi_interp_array_func


def add_detection(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    clin_strat: Stratification,
):
    """Add the process of disease detection to the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        clin_strat: The clinical stratification object
    """
    tv_detection_rate = defer(tanh_based_scaleup)(
        Time,
        Parameter("passive_detection_shape", 0.0),
        Parameter("passive_detection_inflection", 0.0),
        Parameter("passive_detection_past", 0.0),
        Parameter("passive_detection_current", 0.0),
    )
    detect = TransitionFlow(
        "detection",
        (disease_state["active"], clin_strat["clin"]),
        disease_state["treatment"],
        tv_detection_rate,
    )
    epi_model.add_flow(detect)


def compute_outcome_props(
    rx_duration: float,
    prop_neg_rx_death: float,
    tsr: float,
    death_rate: np.array,
) -> Tuple[np.array]:
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
    return success, relapse_prop, prop_death_from_rx


def get_outcome_rate(
    outcome: str,
    rx_duration: float,
    prop_neg_rx_death: float,
    tsr: float,
    death_rate: np.array,
) -> np.array:
    """Get the flow rate for a specific treatment outcome.

    Args:
        outcome: The outcome identifier (success, relapse or rx_death)
        rx_duration: Treatment duration in model time units (years)
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes resulting in death
        tsr: Treatment success rate
        death_rate: Natural death rate

    Returns:
        The flow rate for the outcome requested
    """
    success, relapse_prop, prop_death_from_rx = compute_outcome_props(
        rx_duration,
        prop_neg_rx_death,
        tsr,
        death_rate,
    )
    if outcome == "success":
        result = success
    elif outcome == "relapse":
        result = relapse_prop
    elif outcome == "rx_death":
        result = prop_death_from_rx

    return result / rx_duration


def add_treatment_flows(
    death_rates: pd.DataFrame,
    start_time: float,
    epi_model: CompartmentalModelODE,
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
    death_unsucc_func = defer(lambda t: get_cos_multicurve(t, death_unsucc_times, death_unsucc_vals))(Time)

    # Natural death calculations
    death_array_func = make_multi_interp_array_func(
        death_rates.index.to_numpy(dtype=float),
        death_rates.to_numpy(dtype=float),
        start_time,
    )
    death_func = defer(death_array_func)(Time)

    # Other common variables
    rx_source = (disease_state["treatment"], age_strat[age_strat.strata])
    dur = Parameter("rx_duration", 0.0)

    # Success
    succ_dest = (disease_state["recovered"], age_strat[age_strat.strata])
    succ_rate = defer(get_outcome_rate)("success", dur, death_unsucc_func, tsr_func, death_func)
    succ_flow = TransitionFlow("success", rx_source, succ_dest, succ_rate)
    epi_model.add_flow(succ_flow)

    # Relapse
    rel_dest = (clin_strat["subclin"], infect_strat["low"], age_strat[age_strat.strata])
    relapse_rate = defer(get_outcome_rate)("relapse", dur, death_unsucc_func, tsr_func, death_func)
    relapse_flow = TransitionFlow("relapse", rx_source, rel_dest, relapse_rate)
    epi_model.add_flow(relapse_flow)

    # # Death - broken code
    rx_death_dest = (disease_state["mtb_naive"], age_strat["0"])
    # rx_death_rate = defer(get_outcome_rate)("rx_death", dur, death_unsucc_func, tsr_func, death_func)
    # rx_death_flow = TransitionFlow("rx_death", rx_source, rx_death_dest, rx_death_rate)
    # epi_model.add_flow(rx_death_flow)

    # Death - working code
    for a, age in enumerate(AGE_STRATA):
        death_func_scalar = defer(lambda t, i=a: death_array_func(t)[i])(Time)
        rx_death_rate = defer(get_outcome_rate)(
            "rx_death", dur, death_unsucc_func, tsr_func, death_func_scalar
        )
        epi_model.add_flow(TransitionFlow(f"rx_death_{age}", (disease_state["treatment"], age_strat[str(age)]), rx_death_dest, rx_death_rate))
