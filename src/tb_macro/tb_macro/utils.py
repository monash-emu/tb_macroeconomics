from jax import numpy as jnp


def get_triang_vals(
    time: float,
    peak_time: float,
    peak_height: float,
    width: float,
) -> float:
    """Get a value between 0 and peak_height based on
    a triangular function
    with the specified peak time, height and width.

    Args:
        time: The time at which to evaluate the triangular function
        peak_time: The time at which the triangular function reaches its peak
        peak_height: The height of the triangular function at its peak
        width: The width of the triangular function

    Returns:
        The value of the triangular function at the specified time
    """
    time_from_peak = jnp.absolute(time - peak_time)
    return jnp.clip(peak_height * (1.0 - time_from_peak / width), a_min=0.0)


def tanh_based_scaleup(
    t: float,
    shape: float,
    inflection_time: float,
    start_asymptote: float,
    end_asymptote: float,
) -> float:
    """Get a value between start_asymptote and end_asymptote based on
    a hyperbolic tangent function.

    Args:
        t: The time at which to evaluate the function
        shape: The shape parameter of the hyperbolic tangent function
        inflection_time: The time at which the function reaches its inflection point
        start_asymptote: The value of the function as t approaches negative infinity
        end_asymptote: The value of the function as t approaches positive infinity

    Returns:
        The value of the function at the specified time
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote
