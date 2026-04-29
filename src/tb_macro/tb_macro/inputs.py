from tb_macro.constants import DATA_PATH

import pandas as pd


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
