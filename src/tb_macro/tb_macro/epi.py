from typing import List
from collections import namedtuple
import numpy as np
from jax import numpy as jnp
import pandas as pd
from summer3.epi import (
    TransitionFlow,
    CompartmentalEpiModel,
    CompartmentMap,
    Stratification,
    strat_data_from_pandas,
)
from summer3.graph import defer, Parameter, Time, CompartmentValues
from summer3.epi import Category, CategoryData, ManagedArray, CategoryGroup, StratSpec
from tb_macro.demography import (
    add_replacement_deaths,
    add_ageing_flows,
    add_entry_births,
)
from tb_macro.health_system import add_treatment_flows, add_detection
from summer3.arrayops import mul_ma_catdata
from tb_macro.constants import (
    ALL_COMPARTMENTS,
    AGE_STRATA,
    INF_STRATA,
    INFECT_COMPS,
    START_TIME,
    YOUNG_END_AGE,
    TOP_AGE_BRACKET_INFLATION,
    OUTPUT_TIME_STEP,
)
from tb_macro.utils import get_triang_vals
from tb_macro.mixing import get_norm_c_matrix
from tb_macro.acf import add_acf

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
    times = pd.Index(np.arange(start_time, end_time, OUTPUT_TIME_STEP))
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
    age_strat: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
):
    """Add non-infection-related natural history flows to the epidemiological model.

    Args:
        epi_model: The epidemiological model
        disease_state: The disease state compartments
        age_strat: The age stratification object
        clin_strat: The clinical stratification
        infect_strat: The infectiousness stratification

    Notes:
    -----
    After infection is contained, clearance occurs at rate {{clearance_rate}}
    (prior range {{clearance_rate_low}} to {{clearance_rate_up}}) and endogenous
    reactivation (breakdown) at rate {{breakdown_rate}} (prior range
    {{breakdown_rate_low}} to {{breakdown_rate_up}}).

    Among people with active TB, infectiousness and symptoms can each
    increase or decrease. Infectiousness is gained at rate
    {{infectiousness_gain_rate}} and lost at rate {{infectiousness_loss_rate}};
    clinical disease develops at rate {{clinical_progression_rate}} and
    regresses at rate {{clinical_regression_rate}}.

    Subclinical disease may self-resolve at rate {{self_recovery_rate}}.
    Untreated clinical TB causes death at {{tb_mortality_rate_inf}} per year
    if infectious and {{tb_mortality_rate_lowinf}} if not.
    """
    source = disease_state["contained"]
    dest = disease_state["cleared"]
    rate = Parameter("clearance_rate", 0.0)
    clearance = TransitionFlow("clearance", source, dest, rate)
    epi_model.add_flow(clearance)

    source = disease_state["contained"]
    dest = disease_state["incipient"]
    rate = Parameter("breakdown_rate", 0.0)
    breakdown = TransitionFlow("breakdown", source, dest, rate)
    epi_model.add_flow(breakdown)

    source = infect_strat["low"]
    dest = infect_strat["high"]
    rate = Parameter("infectiousness_gain_rate", 0.0)
    infect_gain = TransitionFlow("infectiousness_gain", source, dest, rate)
    epi_model.add_flow(infect_gain)

    source = infect_strat["high"]
    dest = infect_strat["low"]
    rate = Parameter("infectiousness_loss_rate", 0.0)
    infect_loss = TransitionFlow("infectiousness_loss", source, dest, rate)
    epi_model.add_flow(infect_loss)

    source = clin_strat["subclin"]
    dest = clin_strat["clin"]
    rate = Parameter("clinical_progression_rate", 0.0)
    clin_dev = TransitionFlow("clinical_develop", source, dest, rate)
    epi_model.add_flow(clin_dev)

    source = clin_strat["clin"]
    dest = clin_strat["subclin"]
    rate = Parameter("clinical_regression_rate", 0.0)
    clin_regress = TransitionFlow("clinical_regress", source, dest, rate)
    epi_model.add_flow(clin_regress)

    source = (disease_state["active"], clin_strat["subclin"])
    dest = disease_state["recovered"]
    rate = Parameter("self_recovery_rate", 0.0)
    self_recovery = TransitionFlow("self_recovery", source, dest, rate)
    epi_model.add_flow(self_recovery)

    def mort_rates(low_rate, high_rate):
        return infect_strat.categories().wrap(jnp.array([low_rate, high_rate]))

    source = (disease_state["active"], clin_strat["clin"])
    dest = (disease_state["mtb_naive"], age_strat["0"])
    rate = defer(mort_rates)(
        Parameter("tb_mortality_rate_lowinf", 0.0),
        Parameter("tb_mortality_rate_inf", 0.0),
    )
    tb_mort = TransitionFlow("tb_mortality", source, dest, rate)
    epi_model.add_flow(tb_mort)


def infect_process(
    compartment_values: ManagedArray,
    age_cats: CategoryGroup,
    infectious_compartments: StratSpec,
    infectivity_cats: CategoryGroup,
    clinical_cats: CategoryGroup,
    transmission_rate: float,
    age_breaks: jnp.array,
    young_end_age: int,
    rel_sus_children: float,
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
        transmission_rate: Base contact rate multiplier
        age_breaks: Age values used to determine young-age stratification
        young_end_age: Maximum age to receive reduced susceptibility
        rel_sus_children: Susceptibility multiplier for younger ages
        rel_infect_lowinf: Relative infectiousness for low-infectious cases
        rel_infect_subclin: Relative infectiousness for subclinical cases
        mm_dynamic: Function that builds a mixing matrix at a given time

    Returns:
        CategoryData containing the age-stratified force of infection.
    """
    infectee_cats = age_cats
    infect_pop_cats = age_cats.product(infectious_compartments)

    age_infect = jnp.where(age_breaks < young_end_age, 0.0, 1.0)
    age_suscept = jnp.where(age_breaks < young_end_age, rel_sus_children, 1.0)

    infectivity_modifier = infectivity_cats.wrap(jnp.array([rel_infect_lowinf, 1.0]))
    effective_values = mul_ma_catdata(compartment_values, infectivity_modifier, True)

    clin_modifier = clinical_cats.wrap(jnp.array([rel_infect_subclin, 1.0]))
    effective_values = mul_ma_catdata(effective_values, clin_modifier, True)

    ipops = effective_values.sumcats(infect_pop_cats).data
    total_pop = compartment_values.sumcats(age_cats).data

    inf_pressure = transmission_rate * age_infect * ipops / total_pop
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
    start_time: float,
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
        start_time: Run start time
    """

    dynamic_mm = defer(get_norm_c_matrix)(
        jnp.array(age_weights),
        jnp.array(age_weights.index[[0, -1]]),
        jnp.array(group_popsize),
        jnp.array(group_popsize.index[[0, -1]]),
        jnp.array(fert_padded),
        jnp.array(fert_padded.index[[0, -1]]),
        Time + start_time,
        Parameter("bg_mixing", 0.0),
        Parameter("a_spread", 0.0),
        Parameter("pc_strength", 0.0),
    ).set_name("dynamic_mm")
    for comp in INFECT_COMPS:
        suscept_comp = "cleared" if comp in ["cleared", "recovered"] else comp
        rel_sus = Parameter(f"rel_sus_{suscept_comp}", 0.0)
        scaled_contact_rate = Parameter("raw_transmission_rate", 0.0) * rel_sus
        reinfect_foi = defer(infect_process)(
            CompartmentValues,
            age_strat.categories(),
            disease_state["active"],
            infect_strat.categories(),
            clin_strat.categories(),
            scaled_contact_rate,
            jnp.array(AGE_STRATA),
            young_end_age,
            Parameter("rel_sus_children", 0.0),
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
    start_time: float,
):
    """Add the seeding of infection into the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        start_time: Run start time

    Notes:
    -----
    Infection is seeded from the Mtb-naive compartment into incipient
    infection with a triangular pulse peaking in {{seed_peak_time}}, lasting
    {{seed_duration}} years, at a peak rate of {{seed_peak_rate}}.
    """
    peak_time = Parameter("seed_peak_time", 0.0)
    peak_height = Parameter("seed_peak_rate", 0.0)
    width = Parameter("seed_duration", 0.0)
    source = disease_state["mtb_naive"]
    dest = disease_state["incipient"]
    sim_time = Time + start_time
    rate = defer(get_triang_vals)(sim_time, peak_time, peak_height, width)
    seed_flow = TransitionFlow("seed_peak", source, dest, rate)
    epi_model.add_flow(seed_flow)


def get_latency_age_adj(
    age_strat: Stratification,
) -> callable:
    """Get the function to adjust age groups according to
    our standard latency bands: <5, 5 to <15 and 15+.

    Args:
        age_strat: The age stratification object

    Returns:
        The age latency adjustment function
    """
    idx_0 = [a for a in age_strat.strata if int(a) < 5]
    idx_5 = [a for a in age_strat.strata if 5 <= int(a) < 15]
    idx_15 = [a for a in age_strat.strata if 15 <= int(a)]
    latency_cats = CategoryGroup(
        [Category(age_strat[s]) for s in [idx_0, idx_5, idx_15]]
    )

    def latency_age_adj(p_0, p_5, p_15):
        return latency_cats.wrap(jnp.array([p_0, p_5, p_15]))

    return latency_age_adj


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
    latency_age_adj = get_latency_age_adj(age_strat)

    contain_func = defer(latency_age_adj)(
        Parameter("containment_rate_age0", 0.0),
        Parameter("containment_rate_age5", 0.0),
        Parameter("containment_rate_age15", 0.0),
    )
    source = disease_state["incipient"]
    dest = disease_state["contained"]
    contain = TransitionFlow("containment", source, dest, contain_func)
    epi_model.add_flow(contain)

    def inf_prog_adj(p_inf) -> CategoryData:
        return infect_strat.categories().wrap(jnp.array([1.0 - p_inf, p_inf]))

    prog_func = defer(latency_age_adj)(
        Parameter("progression_rate_age0", 0.0),
        Parameter("progression_rate_age5", 0.0),
        Parameter("progression_rate_age15", 0.0),
    )
    source = disease_state["incipient"]
    dest = clin_strat["subclin"]
    prog = TransitionFlow("progression", source, dest, prog_func)
    prog.adjustments_dest.append(
        defer(inf_prog_adj)(Parameter("progression_prop_infectious", 0.0))
    )
    epi_model.add_flow(prog)


def add_flows_to_model(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    clin_strat: Stratification,
    infect_strat: Stratification,
    age_weights: pd.DataFrame,
    group_popsize: pd.DataFrame,
    fert_padded: pd.DataFrame,
    death_rates: pd.DataFrame,
    tsr: pd.Series,
    death_in_unsucc: pd.Series,
    entry_times: np.array,
    entry_rates: np.array,
):
    """Add the transition flows to the TB model.

    Args:
        epi_model: The epidemiological model
        disease_state: The disease state compartments
        age_strat: The age stratification object
        clin_strat: The clinical stratification
        infect_strat: The infectiousness stratification
        age_weights: The age weights for the mixing matrix
        group_popsize: The population data
        fert_padded: The fertility data for the mixing matrix
        death_rates: The per capita death rates
        tsr: Treatment success rate
        entry_times: Entry years
        entry_rates: Calculated entry rates to match population
    """
    death_rates = death_rates.copy()
    death_rates[AGE_STRATA[-1]] *= TOP_AGE_BRACKET_INFLATION
    add_infection_flows(
        epi_model,
        disease_state,
        age_strat,
        clin_strat,
        infect_strat,
        age_weights,
        group_popsize,
        fert_padded,
        YOUNG_END_AGE,
        START_TIME,
    )
    add_natural_history(epi_model, disease_state, age_strat, clin_strat, infect_strat)
    add_ageing_flows(epi_model, age_strat)
    add_seeding(epi_model, disease_state, START_TIME)
    add_detection(epi_model, disease_state, clin_strat, START_TIME)
    add_replacement_deaths(epi_model, disease_state, age_strat, death_rates, START_TIME)
    add_entry_births(
        epi_model, disease_state, age_strat, START_TIME, entry_rates, entry_times
    )
    add_treatment_flows(
        death_rates,
        START_TIME,
        epi_model,
        disease_state,
        age_strat,
        infect_strat,
        clin_strat,
        tsr,
        death_in_unsucc,
    )
    add_latency_flows(epi_model, disease_state, age_strat, clin_strat, infect_strat)
    add_acf(epi_model, disease_state, START_TIME)



def initialise_pops(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    start_apops: List[float],
):
    """Initialise the model population

    Args:
        epi_model: The epidemiological model
        disease_state: The disease state compartments
        age_strat: The age stratification object
        start_apops: The population distribution by age
    """
    init_apops_series = pd.Series(
        index=[str(a) for a in AGE_STRATA], data=np.array(start_apops)
    )
    init_apops = strat_data_from_pandas(init_apops_series, age_strat)
    init_dpops = [0.0] * len(ALL_COMPARTMENTS)
    init_dpops[ALL_COMPARTMENTS.index("mtb_naive")] = 1.0
    pop_splits = [CategoryData(disease_state.categories(), jnp.array((init_dpops)))]
    epi_model.set_initial_population(init_apops, pop_splits)
    epi_model.computed_values.append("dynamic_mm")
