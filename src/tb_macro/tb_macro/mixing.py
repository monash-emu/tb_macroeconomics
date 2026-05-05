import pandas as pd
from jax import numpy as jnp

from tb_macro.constants import AGE_STRATA, MAX_AGE


def get_year_index(
    ends: pd.DataFrame,
    time: float,
) -> jnp.array:
    """Get the relevant row index from a dataframe
    with consecutive index values representing years
    given a time/year input.

    Args:
        data: The data that may or may not span time
        time: Model time

    Returns:
        The relevant row of data
    """
    clamped_time = jnp.clip(time, ends[0], ends[1])
    return (clamped_time - ends[0]).astype(jnp.int32)


def get_assortative_component(
    ages_i: jnp.array,
    ages_j: jnp.array,
    a_spread: float,
    weight_prod: jnp.array,
) -> float:
    """Population-weighted assortative mixing contribution
    between two age bands.

    Args:
        ages_i: Contributing ages by row
        ages_j: Contributing ages by column
        a_spread: Decay parameter
        weight_prod: Outer product of the weights

    Returns:
        Assortative mixing value
    """
    age_diff_mat = jnp.abs(ages_i[:, None] - ages_j[None, :])
    assort_age_vals = (1.0 / a_spread) * jnp.exp(-age_diff_mat / a_spread)
    return jnp.sum(weight_prod * assort_age_vals)


def get_child_parent_component(ages_i, ages_j, fert, fert_ends, weight_prod, time):
    age_gap_mat = jnp.abs(ages_i[:, None] - ages_j[None, :])
    child_age_mat = jnp.minimum(ages_i[:, None], ages_j[None, :])
    child_birth_years = time - child_age_mat
    clamped_birth_years = get_year_index(fert_ends, child_birth_years)
    return jnp.sum(weight_prod * jnp.array(fert)[clamped_birth_years, age_gap_mat])


def build_s_matrix(
    weights: pd.DataFrame,
    fert: pd.DataFrame,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
) -> jnp.array:
    """Construct the symmetric s_matrix matrix,
    i.e. the per capita, per capita
    matrix that could be used for a density-dependent
    transmission model.

    Args:
        weights: Within age brackets weight by age group and year
        fert: ***
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing

    Returns:
        The s_matrix matrix
    """
    n_groups = len(AGE_STRATA)
    s_matrix = jnp.zeros((n_groups, n_groups))
    weight_ends = weights.index[[0, -1]]
    year_idx = get_year_index(weight_ends, time)
    current_weights = jnp.array(weights)[year_idx, :]

    for i, lower_i in enumerate(AGE_STRATA):
        upper_i = MAX_AGE + 1 if lower_i == AGE_STRATA[-1] else AGE_STRATA[i + 1]
        ages_i = jnp.arange(lower_i, upper_i)
        weights_i = current_weights[lower_i:upper_i]

        for j, lower_j in enumerate(
            AGE_STRATA[: i + 1]
        ):  # compute for lower triangular and diagonal
            upper_j = MAX_AGE + 1 if lower_j == AGE_STRATA[-1] else AGE_STRATA[j + 1]
            ages_j = jnp.arange(lower_j, upper_j)
            weights_j = current_weights[lower_j:upper_j]

            weight_prod = jnp.outer(weights_i, weights_j)

            assort_component = get_assortative_component(
                ages_i, ages_j, a_spread, weight_prod
            )

            child_parent_component = get_child_parent_component(
                ages_i, ages_j, fert, fert_ends, weight_prod, time
            )

            value = bg_mixing + assort_component + child_parent_component
            s_matrix = s_matrix.at[i, j].set(value)
            s_matrix = s_matrix.at[j, i].set(value)
    return s_matrix


def build_c_matrix(
    weights: pd.DataFrame,
    pops: pd.DataFrame,
    fert: pd.DataFrame,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
) -> jnp.array:
    """Get the C matrix, being the per capita
    or frequency-dependent transmission matrix
    from the per capita, per capita or
    density-dependent transmission matrix.
    Note that the [None, :] is not strictly necessary
    but makes it clearer that this is row vector.

    Args:
        weights: Within age brackets weight by age group and year
        pops: Population sizes by age group
        fert: ***
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing

    Returns:
        The C matrix
    """
    pop_ends = pops.index[[0, -1]]
    year_idx = get_year_index(pop_ends, time)
    pops = jnp.array(pops)[year_idx, :]
    return pops[None, :] * build_s_matrix(weights, fert, fert_ends, time, bg_mixing, a_spread)


def get_norm_c_matrix(
    weights: pd.DataFrame,
    pops: pd.DataFrame,
    fert: pd.DataFrame,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
) -> jnp.array:
    """Get the normalised version of the per capita
    mixing matrix created by build_c_matrix.

    Args:
        weights: Within age brackets weight by age group and year
        pops: Population sizes by age group
        fert: ***
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing

    Returns:
        The normalised C matrix
    """
    c_matrix = build_c_matrix(weights, pops, fert, fert_ends, time, bg_mixing, a_spread)
    eigvals = jnp.linalg.eigvals(c_matrix)
    spectral_radius = jnp.max(jnp.abs(eigvals))
    return c_matrix / spectral_radius
