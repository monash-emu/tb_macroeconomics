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
from summer3.graph import defer, Parameter, Time, CompartmentValues
from summer3.epi import CategoryData, ManagedArray, CategoryGroup, StratSpec
from summer3.arrayops import mul_ma_catdata
from tb_macro.constants import ALL_COMPARTMENTS, AGE_STRATA, INF_STRATA, INFECT_COMPS
from tb_macro.utils import get_triang_vals
from tb_macro.mixing import get_norm_c_matrix

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
    infect_strat = Stratification("infectious", INF_STRATA)
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
    epi_model.add_flow(clearance)
    epi_model.add_flow(breakdown)
    epi_model.add_flow(increase_infect)
    epi_model.add_flow(decrease_infect)
    epi_model.add_flow(clin_dev)
    epi_model.add_flow(clin_regress)
    epi_model.add_flow(self_recovery)


def infect_process(
    compartment_values: ManagedArray,
    age_cats: CategoryGroup,
    infectious_compartments: StratSpec,
    infectivity_cats: CategoryGroup,
    clinical_cats: CategoryGroup,
    contact_rate: float,
    freq_dens_exponent: float,
    age_breaks: jnp.array,
    young_end_age: int,
    young_suscept: float,
    rel_infect_lowinf: float,
    rel_infect_subclin: float,
    mm_dynamic,
):
    """Compute the age-specific force of infection.
    Uses compartment values, age structure, mixing and clinical/infectiousness
    modifiers to compute age-stratified force of infection.

    Args:
        compartment_values: Model compartment values across stratifications
        age_cats: Age category group for infectors and infectees
        infectious_compartments: Active disease compartments that contribute to FoI
        infectivity_cats: Category group for infectiousness strata
        clinical_cats: Category group for clinical strata
        contact_rate: Base contact rate multiplier
        freq_dens_exponent: Frequency/density-dependence exponent
        age_breaks: Age values used to determine young-age stratification
        young_end_age: Maximum age to receive reduced susceptibility
        young_suscept: Susceptibility multiplier for younger ages
        rel_infectiousness_lowinf: Relative infectiousness for low-infectious cases
        rel_infectiousness_subclin: Relative infectiousness for subclinical cases
        mm_function: Function that builds a mixing matrix at a given time
        a_spread: Assortative mixing spread parameter
        bg_mixing: Background mixing level
        pc_strength: Parent-child contact strength
        weights: Within-age-group weight matrix
        weight_ends: Start/end indices for weight data
        pops: Population counts by age and time
        pop_ends: Start/end indices for population data
        fert: Fertility data
        fert_ends: Start/end indices for fertility data
        time: Model time at which to compute mixing

    Returns:
        CategoryData containing the age-stratified force of infection.
    """
    infectee_cats = age_cats
    infect_pop_cats = age_cats.product(infectious_compartments)

    age_infect = jnp.where(age_breaks < young_end_age, 0.0, 1.0)
    age_suscept = jnp.where(age_breaks < young_end_age, young_suscept, 1.0)

    infectivity_modifier = infectivity_cats.wrap(jnp.array([rel_infect_lowinf, 1.0]))
    effective_values = mul_ma_catdata(compartment_values, infectivity_modifier)

    clin_modifier = clinical_cats.wrap(jnp.array([rel_infect_subclin, 1.0]))
    effective_values = mul_ma_catdata(effective_values, clin_modifier)

    ipops = effective_values.sumcats(infect_pop_cats).data
    total_pop = compartment_values.sumcats(age_cats).data

    inf_pressure = contact_rate * age_infect * ipops / total_pop**freq_dens_exponent
    age_foi = age_suscept * (mm_dynamic @ inf_pressure)
    return CategoryData(infectee_cats, age_foi)


def add_infection_flows(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
    age_weights: jnp.array,
    group_popsize: jnp.array,
    fert_padded: jnp.array,
    young_end_age: float,
):
    """Add infection flows to the model.
    Flows are added from each susceptible compartment
    to the subclinical compartment,
    with the force of infection computed by the infect_process function.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        clin_strat: The clinical stratification object
        infect_strat: The infectiousness stratification object
        age_weights: The age weights for the mixing matrix
        group_popsize: The population sizes for the mixing matrix
        fert_padded: The fertility data for the mixing matrix
        young_end_age: The maximum age to receive reduced susceptibility
    """
    dynamic_mm = defer(get_norm_c_matrix)(
        jnp.array(age_weights), 
        jnp.array(age_weights.index[[0, -1]]),
        jnp.array(group_popsize),
        jnp.array(group_popsize.index[[0, -1]]),
        jnp.array(fert_padded),
        jnp.array(fert_padded.index[[0, -1]]),
        Time,
        Parameter("bg_mixing", 0.0),
        Parameter("a_spread", 0.0),
        Parameter("pc_strength", 0.0),
    )
    for comp in INFECT_COMPS:
        suscept_comp = "cleared" if comp in ["cleared", "recovered"] else comp
        rel_sus = Parameter(f"rel_sus_{suscept_comp}", 0.0)
        scaled_contact_rate = Parameter("contact_rate", 0.0) * rel_sus
        reinfect_foi = defer(infect_process)(
            CompartmentValues,
            age_strat.categories(),
            disease_state["active"],
            infect_strat.categories(),
            clin_strat.categories(),
            scaled_contact_rate,
            Parameter("freq_dens_exponent", 1.0),
            jnp.array(AGE_STRATA),
            young_end_age,
            Parameter("young_suscept", 0.0),
            Parameter("rel_infectiousness_lowinf", 0.0),
            Parameter("rel_infectiousness_subclin", 0.0),
            dynamic_mm,
        )
        reinfect = TransitionFlow(
            f"infect_{comp}",
            disease_state[comp],
            disease_state["incipient"],
            reinfect_foi,
        )
        epi_model.add_flow(reinfect)


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


def add_latency_flows(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
):
    """Add the latency / infection progression flows to the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        clin_strat: The clinical stratification object
        infect_strat: The infectiousness stratification object
    """
    for age in AGE_STRATA:
        latency_age_cat = "age0" if age < 5 else "age5" if age < 15 else "age15"
        contain = TransitionFlow(
            f"containment_{age}",
            (disease_state["incipient"], age_strat[str(age)]),
            (disease_state["contained"], age_strat[str(age)]),
            Parameter(f"containment_{latency_age_cat}", 0.0),
        )
        epi_model.add_flow(contain)

        for strat in INF_STRATA:
            prop_inf = Parameter("progression_prop_infectious", 0.0)
            strat_prop = prop_inf if strat == "high" else 1.0 - prop_inf
            progression = TransitionFlow(
                f"progression_{strat}_{age}",
                (disease_state["incipient"], age_strat[str(age)]),
                (clin_strat["subclin"], infect_strat[strat], age_strat[str(age)]),
                Parameter(f"progression_{latency_age_cat}", 0.0) * strat_prop,
            )
            epi_model.add_flow(progression)
