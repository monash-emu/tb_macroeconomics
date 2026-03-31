from jax import numpy as jnp

def get_triang_vals(time, peak_time, peak_height, width):
    time_from_peak = jnp.absolute(time - peak_time)
    return jnp.clip(peak_height * (1.0 - time_from_peak / width), a_min=0.0)

def tanh_based_scaleup(t, shape, inflection_time, start_asymptote, end_asymptote):
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote
