from typing import Tuple
import numpy as np
import pandas as pd
from jax import numpy as jnp, vmap

from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow, EntryFlow
from summer3.graph import defer, Time

from tb_macro.constants import AGE_STRATA


def make_multi_interp_func(
    times: np.array,
    rates: np.array,
    start_time: float,
    age_strat: Stratification
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
    def interp_single_age(t, age_rates):
        model_time = t + start_time
        left_val = age_rates[0]
        right_val = age_rates[-1]
        return jnp.interp(model_time, times, age_rates, left=left_val, right=right_val)

    interp_all_ages = vmap(interp_single_age, in_axes=(None, 1)) # No axis for time, columns for age
    return lambda t: age_strat.categories().wrap(interp_all_ages(t, rates))


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
    death_func = make_multi_interp_func(
        death_rates.index.to_numpy(),
        death_rates.to_numpy(),
        start_time,
        age_strat,
    )
    replacement_deaths = TransitionFlow(
        "replacement_deaths",
        (disease_state[disease_state.strata], age_strat[age_strat.strata]),
        (disease_state["mtb_naive"], age_strat["0"]),
        defer(death_func)(Time),
    )
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
    ageing_rates = []
    for a in range(len(AGE_STRATA) - 1):
        bottom = AGE_STRATA[a]
        top = AGE_STRATA[a + 1]
        progression = f"{bottom}_to_{top}"
        ageing_rates.append(1.0 / (top - bottom))

        ageing = TransitionFlow(
            f"ageing_{progression}",
            age_strat[str(bottom)],
            age_strat[str(top)],
            1.0 / (top - bottom),
        )
        epi_model.add_flow(ageing)


def prepare_pop_data_for_entries(
    pop_data: pd.DataFrame,
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
    non_dec_data = pop_data.cummax()
    non_dec_data[start_time] = start_pop
    non_dec_data_w_start = non_dec_data.sort_index()
    entry_birth_rates = (
        non_dec_data_w_start.diff() / non_dec_data_w_start.index.diff()
    ).dropna()
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
    entry_rate = EntryFlow(
        "entry_births",
        (disease_state["mtb_naive"], age_strat["0"]),
        defer(birth_func)(Time),
    )
    epi_model.add_flow(entry_rate)
