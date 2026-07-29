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

# Compartment characteristics
PREV_STATES = [
    "active",
    "treatment",
]
LATENT_STATES = [s for s in ALL_COMPARTMENTS if s != "mtb_naive"]

# Age-related
AGE_STRATA = [0, 10, 20]
MAX_AGE = 120
YOUNG_END_AGE = 15

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"

# Country
ISO3 = "VNM"

# Times
START_TIME = 1800.0
END_TIME = 2101.0

# Calibration
SOLVER_KWARGS = {"max_steps": 4000}
