import pandas as pd

# Data previously obtained from the Vietnam NTP by Long
NOTIF_TARGET = pd.Series(
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
LATENT_TARGET = pd.Series(
    {
        2019: 43.0,
    }
)
PREV_TARGET = pd.Series(
    {
        2017.0: 322.0,
    }
)