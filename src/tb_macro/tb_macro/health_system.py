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

from tb_macro.demography import make_interp_func
from tb_macro.constants import AGE_STRATA
from tb_macro.utils import tanh_based_scaleup, CosineMultiCurve, get_scale_data


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


def get_rx_outcome_rate(
    rx_duration: float,
    prop_neg_rx_death: float,
    time: float,
    death_times: jnp.array,
    death_data: jnp.array,
    start_time: float,
    outcome: str,
    tsr: callable,
) -> float:
    """Get the treatment outcome rate for
    relapse, treatment-related death or success.

    Args:
        rx_duration: Treatment duration
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes
            resulting in death
        time: Model time
        death_times: The per capita death rates
        death_data: The corresponding times for these rates
        start_time: The model starting time as a calendar year
        outcome: The outcome of interest
        tsr: Treatment success "rate" function (returns a proportion)

    Returns:
        The rate value for the outcome of interest
    """
    death_func = make_interp_func(death_times, death_data, start_time)
    death_rates = death_func(time)

    prop_natural_death_on_rx = 1.0 - jnp.exp(-rx_duration * death_rates)
    req_prop_death_on_rx = (1.0 - tsr) * prop_neg_rx_death
    prop_death_from_rx = jnp.maximum(
        req_prop_death_on_rx - prop_natural_death_on_rx, 0.0
    )
    prop_total_death = prop_death_from_rx + prop_natural_death_on_rx

    relapse_prop = jnp.maximum(1.0 - tsr - prop_total_death, 0.0)
    success = jnp.maximum(1.0 - relapse_prop - prop_total_death, 0.0)

    if outcome == "relapse":
        return relapse_prop / rx_duration
    elif outcome == "rx_death":
        return prop_death_from_rx / rx_duration
    elif outcome == "success":
        return success / rx_duration


def add_treatment_flows(
    death_rates: pd.DataFrame,
    start_time: float,
    epi_model: CompartmentalModelODE,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    clin_strat: Stratification,
    tsr_data: pd.DataFrame,
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
    tsr_times = get_scale_data(np.array(tsr_data.index))
    tsr_vals = get_scale_data(np.array(tsr_data))
    interp_func = CosineMultiCurve()
    tsr_func = defer(lambda t: interp_func.get_multicurve(t, tsr_times, tsr_vals))(Time)
    for age in AGE_STRATA:
        rx_source = (disease_state["treatment"], age_strat[str(age)])
        rx_dests = {
            "relapse": (
                clin_strat["subclin"],
                infect_strat["low"],
                age_strat[str(age)],
            ),
            "rx_death": (disease_state["mtb_naive"], age_strat["0"]),
            "success": (disease_state["recovered"], age_strat[str(age)]),
        }
        for outcome in rx_dests:
            outcome_flow = TransitionFlow(
                f"{outcome}_{age}",
                rx_source,
                rx_dests[outcome],
                defer(get_rx_outcome_rate)(
                    Parameter("rx_duration", 0.0),
                    Parameter("prop_neg_rx_death", 0.0),
                    Time,
                    death_rates.index.to_numpy(dtype=float),
                    death_rates[age].to_numpy(dtype=float),
                    start_time,
                    outcome,
                    tsr_func,
                ),
            )
            epi_model.add_flow(outcome_flow)
