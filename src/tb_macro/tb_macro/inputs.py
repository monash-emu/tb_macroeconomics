import re
import numpy as np
import pandas as pd

from tb_macro.constants import BASE_PATH, DATA_PATH, AGE_STRATA, MAX_AGE


def get_country_pop(
    iso3: str,
) -> pd.DataFrame:
    """Get raw UN population data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    data = pd.read_csv(DATA_PATH / "population/un_population_20260506T0211Z.csv")
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


def add_groups_to_single_pop(
    single_age_pops: pd.DataFrame,
) -> pd.DataFrame:
    single_age_pops["Age Group"] = pd.cut(
        single_age_pops["Age"],
        bins=AGE_STRATA + [MAX_AGE],
        labels=AGE_STRATA,
        right=False,
    )


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
    add_groups_to_single_pop(single_age_pops)
    group_popsizes = (
        single_age_pops.groupby(["Time", "Age Group"], observed=True)["Pop"]
        .sum()
        .unstack("Age Group")
    )
    assert np.all(np.diff(group_popsizes.index.values) == 1)
    return group_popsizes


def get_un_mortality(
    iso3: str,
) -> pd.DataFrame:
    """Get UN mortality data for a specified country.

    Args:
        iso3: The country identifier

    Returns:
        Dataframe with columns for age groups and rows for years
    """
    mort_data = pd.read_csv(DATA_PATH / "population/un_mortality_20260506T0212Z.csv")
    relevant_cols = ["Time", "AgeGrp", "DeathTotal"]
    country_filt = mort_data["ISO3_code"] == iso3
    mort_data = mort_data.loc[country_filt, relevant_cols]
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


def lower_conmat(
    age_group: str,
) -> int:
    """Get the lower value of the conmat age band.

    Args:
        age_group: The age group string

    Returns:
        The value
    """
    pattern = r"\[(\d+),"
    return int(re.search(pattern, age_group).group(1))


def load_conmat(
    iso3: str,
) -> pd.DataFrame:
    """Load the conmat data produced by the R script
    and check that age bands match with the modelled ones.

    Args:
        iso3: The country identifier

    Returns:
        Raw conmat data
    """
    conmat_dir = BASE_PATH / "src/tb_macro/tb_macro/conmat/"
    conmat_data = pd.read_csv(conmat_dir / f"conmat_all_{iso3}.csv", index_col=0)
    conmat_agebreaks = [lower_conmat(a) for a in conmat_data["age_group_from"].unique()]
    assert set(AGE_STRATA) == set(
        conmat_agebreaks
    ), "model age bands do not match conmat"
    return conmat_data


def convert_conmat(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the raw conmat format to a square dataframe.

    Args:
        data: Output of load_conmat

    Returns:
        The conmat data as a square matrix
    """
    conmat = data.assign(
        age_from=data["age_group_from"].map(lower_conmat),
        age_to=data["age_group_to"].map(lower_conmat),
    )
    return conmat.pivot(
        index="age_from",
        columns="age_to",
        values="contacts",
    ).reindex(index=AGE_STRATA, columns=AGE_STRATA)


def normalise_spectral_radius(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise matrix by dividing by
    its spectral radius.

    Args:
        matrix: The matrix to normalise

    Returns:
        The normalised matrix
    """
    eigvals = np.linalg.eigvals(matrix)
    spectral_radius = np.max(np.abs(eigvals))
    return matrix / spectral_radius


def build_age_weight_lookup(
    single_age: pd.DataFrame,
) -> pd.DataFrame:
    """Get within age-group weights for each
    single year age of that group.

    Args:
        single_age: The population distribution by age and year

    Returns:
        The within-age group weights
    """
    wide_single_age = single_age.pivot(index="Time", columns="Age", values="Pop")
    weights = pd.DataFrame(index=wide_single_age.index, columns=wide_single_age.columns)
    for a, lower in enumerate(AGE_STRATA):
        upper = MAX_AGE + 1 if lower == AGE_STRATA[-1] else AGE_STRATA[a + 1]
        ages = [age for age in wide_single_age.columns if lower <= age < upper]
        pop_sum = wide_single_age[ages].sum(axis=1).replace(0.0, 1.0)
        weights[ages] = wide_single_age[ages].div(pop_sum, axis=0)
    assert (weights.index.diff()[1:] == 1).all(), "age weight indices not consecutive"
    assert (weights.columns.diff()[1:] == 1).all(), "age weight ages not consecutive"
    return weights


def get_fertility_data(
    iso3: str,
) -> pd.DataFrame:
    """Get the UN fertility data.

    Args:
        iso3: Country identifier

    Returns:
        The data
    """
    filename = f"un_fertility_20260506T0219Z.csv"
    raw_data = pd.read_csv(DATA_PATH / "population" / filename)
    country_data = raw_data.loc[raw_data["ISO3_code"] == iso3]
    data = country_data.pivot(index="Time", columns="AgeGrp", values="ASFR")
    norm_data = data.div(data.sum(axis=1), axis=0)
    norm_data.columns = norm_data.columns.astype(int)
    assert np.all(np.diff(norm_data.index.values) == 1)
    return norm_data


def write_conmat_pop_csv(
    iso3: str,
    year: int,
):
    """Generate the Conmat population data
    needed for mixing matrix construction.

    Args:
        iso3: Country identifier
        year: Year to extract data from
    """
    pop_data = get_country_pop(iso3)
    single_age = get_single_age_pop_from_ungroups(pop_data)
    group_popsize = get_group_popsizes(single_age)
    pops = group_popsize.loc[year]
    pops.index.name = "age"
    pops.name = "population"
    pops.to_csv(BASE_PATH / f"src/tb_macro/tb_macro/conmat/{iso3}_pop_{year}.csv")
