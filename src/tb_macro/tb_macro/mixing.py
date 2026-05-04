import pandas as pd
from jax import numpy as jnp

from tb_macro.constants import AGE_STRATA, MAX_AGE


def get_current_weights(
    weights: pd.DataFrame,
    time: float,
) -> jnp.array:
    """Extract the age weights for the current time
    from the full weights data.

    Args:
        weights: The weights by age and time
        time: Current time being simulated

    Returns:
        A vector for the current weights by age
    """
    start_year = weights.index[0]
    end_year = weights.index[-1]
    clamped_time = jnp.clip(time, start_year, end_year)
    year_idx = (clamped_time - start_year).astype(jnp.int32)
    return jnp.array(weights)[year_idx, :]


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
    """Construct S matrix, i.e. the per capita, per capita
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
    S = jnp.empty((n_groups, n_groups))
    current_weights = get_current_weights(weights, time)

    for i, lower_i in enumerate(AGE_STRATA):
        upper_i = MAX_AGE + 1 if lower_i == AGE_STRATA[-1] else AGE_STRATA[i + 1]
        ages_i = jnp.arange(lower_i, upper_i)
        weights_i = current_weights[lower_i: upper_i]

        for j, lower_j in enumerate(AGE_STRATA):
            upper_j = MAX_AGE + 1 if lower_j == AGE_STRATA[-1] else AGE_STRATA[j + 1]
            ages_j = jnp.arange(lower_j, upper_j)
            weights_j = current_weights[lower_j: upper_j]

            weight_prod = jnp.outer(weights_i, weights_j)

            assort_component = get_assortative_component(ages_i, ages_j, a_spread, weight_prod)

            value = bg_mixing + assort_component
            S = S.at[i, j].set(value)
            S = S.at[j, i].set(value)
    return S


def get_pop(
    pops: pd.DataFrame,
    time: float,
) -> jnp.array:
    """Get current population sizes by modelled
    age group.

    Args:
        pops: Age-group-specific populations for all years
        time: Model time

    Returns:
        Current populations
    """
    start_year = pops.index[0]
    end_year = pops.index[-1]
    clamped_time = jnp.clip(time, start_year, end_year)
    year_idx = (clamped_time - start_year).astype(jnp.int32)
    return jnp.array(pops)[year_idx, :]


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
    pops = get_pop(pops, time)
    s_matrix = build_s_matrix(weights, time, bg_mixing, a_spread)
    return pops[None, :] * s_matrix
