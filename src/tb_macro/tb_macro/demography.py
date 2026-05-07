import pandas as pd
from jax import numpy as jnp

from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow
from summer3.graph import defer, Time

from tb_macro.constants import ALL_COMPARTMENTS, AGE_STRATA


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

    def make_death_func(times, rates):
        def death_func(t):
            model_time = t + start_time
            return jnp.interp(model_time, times, rates, left=rates[0], right=rates[-1])

        return death_func

    for age in AGE_STRATA:
        age_death_rates = death_rates[age]
        times = age_death_rates.index.to_numpy(dtype=float)
        rates = age_death_rates.to_numpy(dtype=float)
        death_func = make_death_func(times, rates)
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
