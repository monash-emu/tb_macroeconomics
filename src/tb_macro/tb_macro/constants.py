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
INFECTED_STATES = [s for s in ALL_COMPARTMENTS if s != "mtb_naive"]

# Age-related
AGE_STRATA = [0, 3, 5, 10, 15, 18, 40, 65]
MAX_AGE = 120
YOUNG_END_AGE = 15
TOP_AGE_BRACKET_INFLATION = 2.0

# Paths
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "outputs"
LOCAL_OUTPUT_PATH = OUTPUT_PATH / "local"

# Country
ISO3 = "VNM"

# Times
START_TIME = 1800.0
END_TIME = 2101.0
CALENDAR_YEAR_MIDPOINT = 0.5
OUTPUT_TIME_STEP = CALENDAR_YEAR_MIDPOINT

# Calibration
COUNT_LOG_SD = 0.1
PROP_LOGIT_SD = 0.2
SOLVER_KWARGS = {"max_steps": 4000}
N_CHAINS_REMOTE = 8
N_RUNS_LOCAL = 40
N_OUTPUT_SAMPLES = 40
SCENARIO_PARAMS = [{}, {"acf_coverage": 0.8, "acf_sensitivity": 0.8}]
