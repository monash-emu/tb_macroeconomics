from jax import numpy as jnp

def get_triang_vals(time, peak_time, peak_height, width):
    time_from_peak = jnp.absolute(time - peak_time)
    return jnp.clip(peak_height * (1.0 - time_from_peak / width), a_min=0.0)
