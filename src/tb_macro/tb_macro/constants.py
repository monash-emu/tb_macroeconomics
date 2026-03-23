# Compartments for TB model
ALL_COMPARTMENTS = [
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "active",
    "treatment",
    "recovered",
]

# Compartments that can be reinfected
INFECT_COMPS = [
    "mtb_naive",
    "contained",
    "cleared",
    "recovered",
]

# Age-related
AGE_STRATA = ["0", "5", "15"]