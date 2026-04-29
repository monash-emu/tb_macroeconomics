from typing import List
import pandas as pd

from tb_macro.constants import DATA_PATH


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
    max_age: int,
) -> pd.DataFrame:
    """Get population data in single year rows
    using UN data from get_country_pop.

    Args:
        data: Output of get_country_pop
        max_age: The highest age to go to for the top age group

    Returns:
        The single-age data
    """
    single_rows = []
    for _, r in data.iterrows():
        pop = r["PopTotal"] * 1000.0
        agegrp = r["AgeGrp"]
        if agegrp.endswith("+"):
            a0 = int(agegrp[:-1])
            a1 = max_age
        else:
            a0, a1 = map(int, str(agegrp).split("-"))

        n_ages = a1 - a0 + 1
        for age in range(a0, a1 + 1):
            single_rows.append({"Time": r["Time"], "Age": age, "Pop": pop / n_ages})

    return pd.DataFrame(single_rows)


def get_group_popsizes(
    single_age_pops: pd.DataFrame,
    age_strata: List[int],
    max_age: int,
) -> pd.DataFrame:
    """Get dataframe for age group populations by year.

    Args:
        single_age_pops: The single age population data from
            get_single_age_pop_from_ungroups
        age_strata: The age groups
        max_age: The highest age to go to for the top age group

    Returns:
        The dataframe with rows for years and columns for age groups
    """
    single_age_pops["Age Group"] = pd.cut(
        single_age_pops["Age"],
        bins=age_strata + [max_age],
        labels=age_strata,
        right=False,
    )
    return (
        single_age_pops.groupby(["Time", "Age Group"], observed=True)["Pop"]
        .sum()
        .unstack("Age Group")
    )
