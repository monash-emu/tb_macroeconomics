import pandas as pd

from tb_macro.constants import CALENDAR_YEAR_MIDPOINT


def _at_midyear(values: dict) -> pd.Series:
    """Place calendar-year observations at mid-year in model time."""
    series = pd.Series(values)
    series.index = series.index.astype(float) + CALENDAR_YEAR_MIDPOINT
    return series


# Data previously obtained from the Vietnam NTP by Long
NOTIF_TARGET = _at_midyear(
    {
        2011: 100518,
        2012: 103906,
        2013: 102196,
        2014: 102087,
        2015: 100780,
        2016: 102097,
        2017: 102725,
        2018: 99658,
        2019: 102503,
        2020: 99582,
        2021: 77657,
        2022: 102479,
        2023: 104517,
    }
)

# Marks et al., Bull WHO
LATENT_TARGET = _at_midyear(
    {
        2016.0: 36.8,
    }
)

# Second prevalence survey, PLOS One
PULM_PREV_TARGET = _at_midyear(
    {
        2017.0: 322.0,
    }
)
INF_PREV_TARGET = _at_midyear(
    {
        2017.0: 79.0 / 322.0,
    }
)

# Nguyen et al., EID
PREV_DECLINE_TARGET = _at_midyear(
    {
        2007.0: 199.0,
        2017.0: 125.0,
    }
)
