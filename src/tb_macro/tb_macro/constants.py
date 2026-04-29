from pathlib import Path

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
AGE_STRATA = [0, 5, 15]
MAX_AGE = 120

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"
