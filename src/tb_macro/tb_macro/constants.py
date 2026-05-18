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

# Compartments that can be infected/reinfected
INFECT_COMPS = [
    "mtb_naive",
    "contained",
    "cleared",
    "recovered",
]

# Infectiousness strata
INF_STRATA = [
    "low",
    "high",
]

# Age-related
AGE_STRATA = [0, 3, 5, 10, 15, 18, 40, 65]
MAX_AGE = 120

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"
