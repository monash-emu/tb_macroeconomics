import pandas as pd
from jax import numpy as jnp


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
