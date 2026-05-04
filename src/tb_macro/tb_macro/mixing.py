import pandas as pd
from jax import numpy as jnp

from tb_macro.constants import AGE_STRATA, MAX_AGE


def get_year_index(
    data: pd.DataFrame,
    time: float,
) -> jnp.array:
    """Get the relevant row from a dataframe
    with consecutive index values representing years
    given a time/year input.

    Args:
        data: The data that may or may not span time
        time: Model time

    Returns:
        The relevant row of data
    """
    start_year = data.index[0]
    end_year = data.index[-1]
    clamped_time = jnp.clip(time, start_year, end_year)
    return (clamped_time - start_year).astype(jnp.int32)


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


def build_s_matrix(
    weights: pd.DataFrame, 
    time: float,
    bg_mixing: float,
    a_spread: float,
) -> jnp.array:
    """Construct the symmetric S matrix, 
    i.e. the per capita, per capita
    matrix that could be used for a density-dependent
    transmission model.

    Args:
        weights: Within age brackets weight by age group and year
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing

    Returns:
        The S matrix
    """
    n_groups = len(AGE_STRATA)
    S = jnp.zeros((n_groups, n_groups))
    year_idx = get_year_index(weights, time)
    current_weights = jnp.array(weights)[year_idx, :]

    for i, lower_i in enumerate(AGE_STRATA):
        upper_i = MAX_AGE + 1 if lower_i == AGE_STRATA[-1] else AGE_STRATA[i + 1]
        ages_i = jnp.arange(lower_i, upper_i)
        weights_i = current_weights[lower_i: upper_i]

        for j, lower_j in enumerate(AGE_STRATA[:i + 1]): # compute for lower triangular and diagonal
            upper_j = MAX_AGE + 1 if lower_j == AGE_STRATA[-1] else AGE_STRATA[j + 1]
            ages_j = jnp.arange(lower_j, upper_j)
            weights_j = current_weights[lower_j: upper_j]

            weight_prod = jnp.outer(weights_i, weights_j)

            assort_component = get_assortative_component(ages_i, ages_j, a_spread, weight_prod)

            value = bg_mixing + assort_component
            S = S.at[i, j].set(value)
            S = S.at[j, i].set(value)
    return S


def build_c_matrix(
    weights: pd.DataFrame,
    pops: pd.DataFrame,
    time: float,
    bg_mixing: float,
    a_spread: float,
) -> jnp.array:
    """Get the C matrix, being the per capita
    or frequency-dependent transmission matrix
    from the per capita, per capita or 
    density-dependent transmission matrix.

    Args:
        weights: Within age brackets weight by age group and year
        pops: Population sizes by age group
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing

    Returns:
        The C matrix
    """
    year_idx = get_year_index(pops, time)
    pops = jnp.array(pops)[year_idx, :]
    s_matrix = build_s_matrix(weights, time, bg_mixing, a_spread)
    return pops[None, :] * s_matrix
