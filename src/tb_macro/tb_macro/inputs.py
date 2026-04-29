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


def get_bounds_from_agegroups(
    lower_bounds: List[int],
    max_age: int,
) -> List[List[int]]:
    """Get the upper and lower bounds for each
    age group being considered in the model.

    Args:
        groups: The lower bounds for the age groups
        max_age: The upper limit for the oldest age group

    Returns:
        List with elements for each age group,
            each being a list containing the upper and lower bounds
    """
    upper_bounds = [a - 1 for a in lower_bounds[1:]] + [max_age]
    return list(zip(lower_bounds, upper_bounds))
