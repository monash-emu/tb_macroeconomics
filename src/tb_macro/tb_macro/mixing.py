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
    weights_array = jnp.array(weights)
    start_year = weights.index[0]
    end_year = weights.index[-1]
    clamped_time = jnp.clip(time, start_year, end_year)  # clamp to data range
    year_idx = (clamped_time - start_year).astype(jnp.int32)  # convert to relative
    return weights_array[year_idx, :]
