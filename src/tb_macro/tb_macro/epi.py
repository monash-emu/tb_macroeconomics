from typing import Optional
import numpy as np
from summer3.epi import CategoryGroup, CategoryData, \
    StratSpec, ManagedArray, mixing_matrix


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
    