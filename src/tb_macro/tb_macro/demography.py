from typing import Tuple
import numpy as np
import pandas as pd
from jax import numpy as jnp

from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow, EntryFlow
from summer3.graph import defer, Time

from tb_macro.constants import ALL_COMPARTMENTS, AGE_STRATA


def make_death_func(
    times: np.array,
    rates: np.array,
    start_time: float,
) -> callable:
    """Make the function of death rates over time
    given the data and the model's starting time.

    Args:
        times: The per capita death rates
        rates: The corresponding times for these rates
        start_time: The model starting time as a calendar year

    Returns:
        The function to get the death rate for a given calendar time
    """
    def death_func(t):
        model_time = t + start_time
        return jnp.interp(model_time, times, rates, left=rates[0], right=rates[-1])

    return death_func


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
    for age in AGE_STRATA:
        age_death_rates = death_rates[age]
        times = age_death_rates.index.to_numpy(dtype=float)
        rates = age_death_rates.to_numpy(dtype=float)
        death_func = make_death_func(times, rates, start_time)
        for comp in ALL_COMPARTMENTS:
            replacement_deaths = TransitionFlow(
                f"replacement_deaths_{comp}_{age}",
                (disease_state[str(comp)], age_strat[str(age)]),
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
    entry_birth_rates = (non_dec_data_w_start.diff() / non_dec_data_w_start.index.diff()).dropna()
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
