from typing import Optional
import numpy as np
from summer3.epi import CategoryGroup, CategoryData, \
    StratSpec, ManagedArray, mixing_matrix, TransitionFlow
from summer3.graph import defer, CompartmentValues, Parameter
from tb_macro.constants import INFECT_COMPS


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

    def process(self, compartment_values: ManagedArray, contact_rate: float, freq_dens_exponent: float):
        ipops = compartment_values.sumcats(self._infectious_pop_cats)
        total_pop = compartment_values.sumcats(self.infector_cats)
        age_foi = (self.mm.data @ (ipops.data / total_pop.data ** freq_dens_exponent)) * contact_rate
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
    contact_rate = Parameter("contact_rate", 0.2)
    freq_dens_exponent = Parameter("freq_dens_exponent", 1.0)
    for comp in INFECT_COMPS:
        reinfect_contact_rate = contact_rate * Parameter(f"rel_sus_{comp}", 1.0)
        reinfect_foi = defer(InfectionProcess.process)(iprocess, CompartmentValues, reinfect_contact_rate, freq_dens_exponent)
        reinfect = TransitionFlow(
            f"infect_{comp}", 
            disease_state[comp], 
            disease_state["incipient"], 
            reinfect_foi,
        )
        epi_model.add_flow(reinfect)
