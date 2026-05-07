from typing import Optional
from collections import namedtuple
import numpy as np
from jax import numpy as jnp
import pandas as pd
from summer3.epi import (
    TransitionFlow,
    CompartmentalEpiModel,
    CompartmentMap,
    Stratification,
)
from summer3.graph import defer, CompartmentValues, Parameter, Time
from tb_macro.constants import ALL_COMPARTMENTS, AGE_STRATA
from tb_macro.utils import get_triang_vals, tanh_based_scaleup

ModelSpec = namedtuple(
    "ModelSpec",
    ["epi_model", "disease_state", "age_strat", "clin_strat", "infect_strat"],
)


def get_base_model(
    start_time: float,
    end_time: float,
) -> ModelSpec:
    """Build and return the base model along with the stratifications.
    Args:
        start_time: Run start time
        end_time: Run end time

    Returns:
        The model, the compartmental states, the age states,
            the clinical states of the active compartment and
            the infectiousness states of the active compartment
    """
    disease_state = Stratification("disease_state", ALL_COMPARTMENTS)
    humans = CompartmentMap.new(disease_state)
    age_strings = [str(a) for a in AGE_STRATA]
    age_strat = humans.stratify(Stratification("age", age_strings))
    infect_strat = Stratification("infectious", ["low", "high"])
    humans.stratify(infect_strat, (disease_state, ["active"]))
    clin_strat = Stratification("clinical", ["subclin", "clin"])
    humans.stratify(clin_strat, (disease_state, ["active"]))
    times = pd.Index(np.arange(start_time, end_time, 1.0))
    return ModelSpec(
        CompartmentalEpiModel(humans, times),
        disease_state,
        age_strat,
        clin_strat,
        infect_strat,
    )


def add_natural_history(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
):
    """Add non-infection-related natural history flows to the epidemiological model.

    Args:
        epi_model: The epidemiological model
        disease_state: The disease state compartments
        clin_strat: The clinical stratification
        infect_strat: The infectiousness stratification
    """
    contain = TransitionFlow(
        "containment",
        disease_state["incipient"],
        disease_state["contained"],
        Parameter("contain", 0.0),
    )
    clearance = TransitionFlow(
        "clearance",
        disease_state["contained"],
        disease_state["cleared"],
        Parameter("clearance_rate", 0.0),
    )
    breakdown = TransitionFlow(
        "breakdown",
        disease_state["contained"],
        disease_state["incipient"],
        Parameter("breakdown_rate", 0.0),
    )
    progression = TransitionFlow(
        "progression",
        disease_state["incipient"],
        clin_strat["subclin"],
        Parameter("progression", 0.0),
    )
    increase_infect = TransitionFlow(
        "increase_infectiousness",
        infect_strat["low"],
        infect_strat["high"],
        Parameter("increase_infect", 0.0),
    )
    decrease_infect = TransitionFlow(
        "decrease_infectiousness",
        infect_strat["high"],
        infect_strat["low"],
        Parameter("decrease_infect", 0.0),
    )
    clin_dev = TransitionFlow(
        "clinical_develop",
        clin_strat["subclin"],
        clin_strat["clin"],
        Parameter("clinical_development", 0.0),
    )
    clin_regress = TransitionFlow(
        "clinical_regress",
        clin_strat["clin"],
        clin_strat["subclin"],
        Parameter("clinical_regression", 0.0),
    )
    self_recovery = TransitionFlow(
        "self_recovery",
        (disease_state["active"], clin_strat["subclin"]),
        disease_state["recovered"],
        Parameter("self_recovery", 0.0),
    )

    # Add flows to model
    epi_model.add_flow(contain)
    epi_model.add_flow(clearance)
    epi_model.add_flow(breakdown)
    epi_model.add_flow(progression)
    epi_model.add_flow(increase_infect)
    epi_model.add_flow(decrease_infect)
    epi_model.add_flow(clin_dev)
    epi_model.add_flow(clin_regress)
    epi_model.add_flow(self_recovery)


def add_health_system_flows(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
):
    """Add the health system-related flows to the epidemiological model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        clin_strat: The clinical stratification object
        infect_strat: The infectiousness stratification object
    """
    treat_recover = TransitionFlow(
        "treatment_recovery",
        disease_state["treatment"],
        disease_state["recovered"],
        Parameter("treatment_recovery", 0.0),
    )
    treat_relapse = TransitionFlow(
        "treatment_relapse",
        disease_state["treatment"],
        (clin_strat["subclin"], infect_strat["low"]),
        Parameter("treatment_relapse", 0.0),
    )
    epi_model.add_flow(treat_recover)
    epi_model.add_flow(treat_relapse)


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


def add_seeding(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
):
    """Add the seeding of infection into the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
    """
    peak_time = Parameter("seed_peak_time", 0.0)
    peak_height = Parameter("seed_peak_rate", 0.0)
    width = Parameter("seed_duration", 0.0)
    seed_flow = TransitionFlow(
        "seed_peak",
        disease_state["mtb_naive"],
        disease_state["incipient"],
        defer(get_triang_vals)(Time, peak_time, peak_height, width),
    )
    epi_model.add_flow(seed_flow)


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
    tv_detection_rate = Parameter("recent_detection_rate", 0.0) * defer(
        tanh_based_scaleup
    )(
        Time,
        Parameter("passive_detection_shape", 0.0),
        Parameter("passive_detection_inflection", 0.0),
        Parameter("passive_detection_past_frac", 0.0),
        1.0,
    )
    detect = TransitionFlow(
        "detection",
        (disease_state["active"], clin_strat["clin"]),
        disease_state["treatment"],
        tv_detection_rate,
    )
    epi_model.add_flow(detect)


def add_replacement_deaths(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    death_rates: pd.DataFrame,
):
    """Add a transition to represent deaths 
    being replaced by births.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        death_rates: The per capita death rates
    """

    def make_death_func(times, rates):
        def death_func(t):
            return jnp.interp(t, times, rates, left=rates[0], right=rates[-1])
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
