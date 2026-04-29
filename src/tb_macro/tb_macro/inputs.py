from typing import List
import pandas as pd

from tb_macro.constants import DATA_PATH, AGE_STRATA, MAX_AGE


def get_country_pop(
    iso3: str,
) -> pd.DataFrame:
    """Get raw UN population data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    data = pd.read_csv(DATA_PATH / "population/un_population.csv")
    return data[data["ISO3_code"] == iso3][["Time", "AgeGrp", "PopTotal"]]


def get_single_age_pop_from_ungroups(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Get population data in single year rows
    using UN data from get_country_pop.

    Args:
        data: Output of get_country_pop

    Returns:
        The single-age data
    """
    single_rows = []
    for _, r in data.iterrows():
        pop = r["PopTotal"] * 1000.0
        agegrp = r["AgeGrp"]
        if agegrp.endswith("+"):
            a0 = int(agegrp[:-1])
            a1 = MAX_AGE
        else:
            a0, a1 = map(int, str(agegrp).split("-"))

        n_ages = a1 - a0 + 1
        for age in range(a0, a1 + 1):
            single_rows.append({"Time": r["Time"], "Age": age, "Pop": pop / n_ages})

    return pd.DataFrame(single_rows)


def get_group_popsizes(
    single_age_pops: pd.DataFrame,
) -> pd.DataFrame:
    """Get dataframe for age group populations by year.

    Args:
        single_age_pops: The single age population data from
            get_single_age_pop_from_ungroups

    Returns:
        The dataframe with rows for years and columns for age groups
    """
    single_age_pops["Age Group"] = pd.cut(
        single_age_pops["Age"],
        bins=AGE_STRATA + [MAX_AGE],
        labels=AGE_STRATA,
        right=False,
    )
    return (
        single_age_pops.groupby(["Time", "Age Group"], observed=True)["Pop"]
        .sum()
        .unstack("Age Group")
    )


def get_un_mortality(
    iso3: str,
    start_year: int,
) -> pd.DataFrame:
    """Get UN mortality data for a country after
    a specified year.

    Args:
        iso3: The country identifier
        start_year: The year to start the data from

    Returns:
        Dataframe with columns for age groups and rows for years
    """
    mort_data = pd.read_csv(DATA_PATH / "population/un_mortality.csv")
    relevant_cols = ["Time", "AgeGrp", "DeathTotal"]
    country_filt = mort_data["ISO3_code"] == iso3
    time_filt = mort_data["Time"] > start_year
    mort_data = mort_data.loc[country_filt & time_filt, relevant_cols]
    mort_data["DeathTotal"] *= 1000.0  # convert from thousands
    mort_data["age"] = mort_data["AgeGrp"].str.replace("100+", "100").astype(int)
    mort_data["Age Group"] = pd.cut(
        mort_data["age"],
        bins=AGE_STRATA + [MAX_AGE],
        labels=AGE_STRATA,
        right=False,
    )
    mort_data = mort_data.groupby(["Time", "Age Group"], as_index=False).sum()
    return mort_data.pivot(index="Time", columns="Age Group", values="DeathTotal")
