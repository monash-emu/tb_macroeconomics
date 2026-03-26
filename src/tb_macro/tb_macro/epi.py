from typing import Optional
from collections import namedtuple
import numpy as np
import pandas as pd
from summer3.epi import (
    CategoryGroup,
    CategoryData,
    StratSpec,
    ManagedArray,
    mixing_matrix,
    TransitionFlow,
    CompartmentalEpiModel,
    CompartmentMap,
    Stratification,
)
from summer3.graph import defer, CompartmentValues, Parameter
from tb_macro.constants import ALL_COMPARTMENTS, INFECT_COMPS, AGE_STRATA


ModelSpec = namedtuple(
    "ModelSpec",
    ["epi_model", "disease_state", "age_strat", "clin_strat", "infect_strat"],
)


def get_base_model():
    disease_state = Stratification("disease_state", ALL_COMPARTMENTS)
    humans = CompartmentMap.new(disease_state)
    age_strings = [str(a) for a in AGE_STRATA]
    age_strat = humans.stratify(Stratification("age", age_strings))
    infect_strat = Stratification("infectious", ["low", "high"])
    humans.stratify(infect_strat, (disease_state, ["active"]))
    clin_strat = Stratification("clinical", ["subclin", "clin"])
    humans.stratify(clin_strat, (disease_state, ["active"]))
    times = pd.Index(np.arange(1800.0, 2000.0, 1.0))
    return ModelSpec(
        CompartmentalEpiModel(humans, times),
        disease_state,
        age_strat,
        clin_strat,
        infect_strat,
    )


class InfectionProcess:
    def __init__(
        self,
        infectee_cats: CategoryGroup,
        infector_cats: CategoryGroup,
        infectious_compartments: StratSpec,
        mm: Optional[ManagedArray] = None,
    ):
        if mm is None:
            mm = mixing_matrix(
                np.ones((len(infectee_cats), len(infector_cats))),
                infector_cats,
                infectee_cats,
            )
        self.mm = mm
        self.infector_cats = infector_cats
        self.infectee_cats = infectee_cats
        self.infectious_compartments = infectious_compartments
        self._infectious_pop_cats = self.infector_cats.product(infectious_compartments)

    def process(
        self,
        compartment_values: ManagedArray,
        contact_rate: float,
        freq_dens_exponent: float,
    ):
        ipops = compartment_values.sumcats(self._infectious_pop_cats)
        total_pop = compartment_values.sumcats(self.infector_cats)
        age_foi = (
            self.mm.data @ (ipops.data / total_pop.data**freq_dens_exponent)
        ) * contact_rate
        return CategoryData(self.infectee_cats, age_foi)


def add_infection_flows(epi_model, disease_state, age_cats):
    """Add the infection-related flows to the epidemiological model.
    This includes both first infections and reinfections,
    with the same approach to calculation, but potentially different contact rates
    base on susceptibility.

    Args:
        epi_model: The epidemiological model
        disease_state: The disease state compartments
        age_cats: The age groups
    """
    iprocess = defer(InfectionProcess)(age_cats, age_cats, disease_state["active"])
    contact_rate = Parameter("contact_rate", 0.0)
    freq_dens_exponent = Parameter("freq_dens_exponent", 1.0)
    for comp in INFECT_COMPS:
        reinfect_contact_rate = contact_rate * Parameter(f"rel_sus_{comp}", 1.0)
        reinfect_foi = defer(InfectionProcess.process)(
            iprocess, CompartmentValues, reinfect_contact_rate, freq_dens_exponent
        )
        reinfect = TransitionFlow(
            f"infect_{comp}",
            disease_state[comp],
            disease_state["incipient"],
            reinfect_foi,
        )
        epi_model.add_flow(reinfect)


def add_transition_flows(epi_model, disease_state, clin_strat, infect_strat):
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


def add_health_system_flows(epi_model, disease_state, clin_strat, infect_strat):
    """Add the health system-related flows to the epidemiological model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        clin_strat: The clinical stratification object
        infect_strat: The infectiousness stratification object
    """
    detect = TransitionFlow(
        "passive_detection",
        disease_state["active"],
        disease_state["treatment"],
        Parameter("detection", 0.0),
    )
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
    epi_model.add_flow(detect)
    epi_model.add_flow(treat_recover)
    epi_model.add_flow(treat_relapse)


def add_ageing_flows(epi_model, age_strat):
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
