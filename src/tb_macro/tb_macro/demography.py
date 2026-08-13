from typing import Tuple
import numpy as np
import pandas as pd
from jax import numpy as jnp

from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow, EntryFlow
from summer3.graph import defer, Time

from tb_macro.constants import AGE_STRATA


def make_single_interp_func(
    times: jnp.ndarray,
    rates: jnp.ndarray,
    start_time: float,
) -> callable:
    """Create a single-series interpolation function."""

    def interp(t):
        sim_time = t + start_time
        return jnp.interp(
            sim_time,
            times,
            rates,
            left=rates[0],
            right=rates[-1],
        )

    return interp


def make_multi_interp_array_func(
    times: jnp.ndarray,
    rates: jnp.ndarray,
    start_time: float,
) -> callable:

    age_funcs = [
        make_single_interp_func(times, rates[:, i], start_time)
        for i in range(rates.shape[1])
    ]

    def interp_all_ages(t):
        return jnp.array([f(t) for f in age_funcs])

    return interp_all_ages


def make_multi_interp_func(
    times: np.array, rates: np.array, start_time: float, age_strat: Stratification
) -> callable:
    """Create a function that can return a vector of values
    pertaining to each modelled age group given some input data
    structured by time and by age group.

    Args:
        times: Time values to use for interpolation
        rates: Rate values to interpolate, repated for each age group
        start_time: Model start time
        age_strat: The age stratification object

    Returns:
        The function to return a vector of values for each age group at a given time
    """
    base_func = make_multi_interp_array_func(times, rates, start_time)
    return lambda t: age_strat.categories().wrap(base_func(t))


def add_replacement_deaths(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    death_rates: pd.DataFrame,
    start_time: float,
):
    """Add a transition to represent deaths
    being replaced by births.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        death_rates: The per capita death rates
        start_time: Model start time
    """
    death_times = np.array(death_rates.index)
    dest = (disease_state["mtb_naive"], age_strat["0"])
    for age in AGE_STRATA:
        age_str = str(age)
        rates = death_rates[age].to_numpy()
        source = age_strat[age_str]
        death_func = make_single_interp_func(death_times, rates, start_time)
        death_name = f"nontb_deaths_{age_str}"
        replacement_deaths = TransitionFlow(death_name, source, dest, defer(death_func)(Time))
        epi_model.add_flow(replacement_deaths)


def add_ageing_flows(
    epi_model: CompartmentalEpiModel,
    age_strat: Stratification,
):
    """Add ageing transition flows between age strata in the epidemiological model.
    Creates and adds TransitionFlow objects to the model that represent
    the progression of the population through sequential age groups.

    Args:
        epi_model: The epidemiological model to add the flows to
        age_strat: The age stratification object
    """
    for a in range(len(AGE_STRATA) - 1):
        lower = AGE_STRATA[a]
        upper = AGE_STRATA[a + 1]
        progression = f"{lower}_to_{upper}"
        rate = 1.0 / (upper - lower)
        l_strat = age_strat[str(lower)]
        u_strat = age_strat[str(upper)]
        ageing = TransitionFlow(f"ageing_{progression}", l_strat, u_strat, rate)
        epi_model.add_flow(ageing)


def prepare_pop_data_for_entries(
    group_popsize: pd.DataFrame,
    start_time: float,
    start_pop: float,
) -> Tuple[jnp.array]:
    """Prepare the aggregate total population data
    for use by the model for new entries.

    Args:
        pop_data: The population data
        start_time: Model start time
        start_pop: Model starting population

    Returns:
        The times and entry rates
    """
    total_pop_size = group_popsize.sum(axis=1)
    # non_dec_data = total_pop_size.cummax()
    total_pop_size[start_time] = start_pop
    non_dec_data_w_start = total_pop_size.sort_index()
    pop_increments = non_dec_data_w_start.diff()
    time_increments = non_dec_data_w_start.index.diff()
    entry_birth_rates = (pop_increments / time_increments).dropna()
    times = jnp.array(entry_birth_rates.index)
    rates = jnp.array(entry_birth_rates)
    return times, rates


def get_birth_rate_func(
    start_time: float,
    rates: jnp.array,
    times: jnp.array,
) -> callable:
    """Get the birth rate function for use by the
    model in

    Args:
        start_time: Model start time
        rates: Birth entry rates
        times: Corresponding times for entry rates

    Returns:
        The birth rate function
    """

    def birth_rate_func(model_time):
        time = model_time + start_time
        idx = jnp.searchsorted(times, time)
        return rates[idx]

    return birth_rate_func


def add_entry_births(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    start_time: float,
    rates: jnp.array,
    times: jnp.array,
):
    """Add entry births to a previously
    closed population model to match a target
    population size over time.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        start_time: Model start time
        rates: Birth entry rates
        times: Corresponding times for entry rates
    """
    birth_func = get_birth_rate_func(start_time, rates, times)
    dest = (disease_state["mtb_naive"], age_strat["0"])
    entry_rate = EntryFlow("entry_births", dest, defer(birth_func)(Time))
    epi_model.add_flow(entry_rate)
