from jax import numpy as jnp
import pandas as pd
import numpy as np

from summer3.epi import (
    TransitionFlow,
    Stratification,
    CompartmentalEpiModel,
)
from summer3.graph import defer, Time, Parameter

from tb_macro.utils import get_scale_data, get_cos_multicurve, get_four_element_multicurve
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

    Notes:
    -----
    Routine detection moves people with clinical active TB into
    treatment. Subclinical disease is not detected by this
    process.

    The rate of detection remains zero until 1957 
    and then follows a cosine-smoothed
    scale-up through 1986 and 2010 to the "{{detect_rate_current}}"
    in 2020. The 2010 rate is the current rate multiplied by the
    "{{rel_detect_2010}}", and the 1986 rate is that 2010 rate
    multiplied by the "{{rel_detect_1986}}".
    """
    detect_rate_2020 = Parameter("detect_rate_current", 0.0)
    detect_rate_2010 = detect_rate_2020 * Parameter("rel_detect_2010", 0.0)
    detect_rate_1986 = detect_rate_2010 * Parameter("rel_detect_1986", 0.0)
    detect_rate_1957 = 0.0

    sim_time = Time + start_time
    detect_func = defer(get_four_element_multicurve)(
        sim_time,
        1957.0,
        detect_rate_1957,
        1986.0,
        detect_rate_1986,
        2010.0,
        detect_rate_2010,
        2020.0,
        detect_rate_2020,
    )

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
    r"""Get the numeric values for all the treatment outcomes.

    Args:
        rx_duration: Treatment duration in model time units (years)
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes resulting in death
        tsr: Treatment success rate
        death_rate: Natural death rate

    Returns:
        Treatment outcome proportions for each of the three outcomes

    Notes:
    -----
    The probability of background death during a course of
    treatment is $1 - \exp(-\delta \mu)$, where $\delta$ is the
    "{{rx_duration}}" and $\mu$ is the age-specific background
    mortality rate.

    The remaining outcomes are split using the treatment success
    rate and the proportion of unsuccessful outcomes that are
    deaths. Background deaths already counted are subtracted from
    that death target, so that only the additional deaths 
    during treatment in excess of background mortality
    are attributed to the TB-related mortality transition.
    Whatever is left of the unsuccessful fraction 
    after these deaths is considered as relapse.
    Success is calculated as the complement of 
    treatment-related death and relapse.
    """
    prop_nat_death_on_rx = 1.0 - jnp.exp(-rx_duration * death_rate)
    req_prop_death_on_rx = (1.0 - tsr) * prop_neg_rx_death
    prop_death_from_rx = jnp.maximum(req_prop_death_on_rx - prop_nat_death_on_rx, 0.0)
    prop_total_death = prop_death_from_rx + prop_nat_death_on_rx
    relapse_prop = jnp.maximum(1.0 - tsr - prop_total_death, 0.0)
    success = jnp.maximum(1.0 - relapse_prop - prop_total_death, 0.0)
    return {"success": success, "relapse": relapse_prop, "rx_death": prop_death_from_rx}


def get_outcome_rates(
    dur: float,
    prop_neg_rx_death: float,
    tsr: float,
    death_rate: np.array,
    age_strat,
) -> np.array:
    """Get the flow rates for the treatment outcomes.

    Args:
        dur: Treatment duration in model time units (years)
        prop_neg_rx_death: Proportion of unsuccessful treatment outcomes resulting in death
        tsr: Treatment success rate
        death_rate: Natural death rate
        age_strat: The age stratification object

    Returns:
        Flow rates for success, relapse and death during treatment

    Notes:
    -----
    Each treatment outcome proportion is converted to a
    competing hazard by dividing by the treatment duration.
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
        tsr_data: Treatment success rate by calendar time
        death_in_unsucc_data: Proportion of unsuccessful outcomes that are deaths,
            by calendar time

    Notes:
    -----
    People on treatment leave to one of three outcomes, with
    rates obtained from the treatment success rate, the
    proportion of unsuccessful outcomes that are deaths, and
    background mortality, all interpolated over calendar time.
    The "{{rx_duration}}" sets the time scale of these rates.

    Success returns people to the recovered compartment.
    Relapse returns them to subclinical, low-infectious active TB.
    Each death during treatment is replaced by an _Mtb_-naive birth 
    into the youngest age group.
    """

    # TSR calculations
    tsr_times = get_scale_data(np.array(tsr_data.index))
    tsr_vals = get_scale_data(np.array(tsr_data))
    sim_time = Time + start_time
    tsr_func = defer(lambda t: get_cos_multicurve(t, tsr_times, tsr_vals))(sim_time)

    # Death in unsuccessful outcomes calculations
    death_unsucc_times = get_scale_data(np.array(death_in_unsucc_data.index))
    death_unsucc_vals = get_scale_data(np.array(death_in_unsucc_data))

    def death_unsucc_curve(t):
        return get_cos_multicurve(t, death_unsucc_times, death_unsucc_vals)

    death_unsucc_func = defer(death_unsucc_curve)(sim_time)

    # Natural death calculations
    death_times = np.array(
        death_rates.index
    )  # FIXME: Does this this need get_scale_data
    death_vals = np.array(death_rates)
    death_array_func = make_multi_interp_array_func(death_times, death_vals, start_time)
    death_func = defer(death_array_func)(Time)

    # Other common variables
    all_age_strata = age_strat[age_strat.strata]
    source = (disease_state["treatment"], all_age_strata)

    # Get all outcome rates
    dur = Parameter("rx_duration", 0.0)
    out_rates = defer(get_outcome_rates)(
        dur, death_unsucc_func, tsr_func, death_func, age_strat
    )

    # Success
    dest = (disease_state["recovered"], all_age_strata)
    flow = TransitionFlow("success", source, dest, out_rates["success"])
    epi_model.add_flow(flow)

    # Relapse
    dest = (clin_strat["subclin"], infect_strat["low"], all_age_strata)
    flow = TransitionFlow("relapse", source, dest, out_rates["relapse"])
    epi_model.add_flow(flow)

    # Death on treatment
    dest = (disease_state["mtb_naive"], age_strat["0"])
    flow = TransitionFlow("rx_death", source, dest, out_rates["rx_death"])
    epi_model.add_flow(flow)
