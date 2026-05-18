from jax import numpy as jnp
import jax
from jax import lax


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


def binary_search_sum_ge(x: float, points: jax.Array) -> int:
    """Return the equivalent of
        (x>=points).sum()
    using a binary search

    Args:
        x: Value to find
        points: Array to search

    Returns:
        (x >= points).sum()
    """

    def cond(state):
        low, high = state
        return (high - low) > 1

    def body(state):
        low, high = state
        midpoint = ((0.5 * (low + high))).astype(int)
        update_upper = x < points[midpoint]
        low = jnp.where(update_upper, low, midpoint)
        high = jnp.where(update_upper, midpoint, high)
        return (low, high)

    low, high = lax.while_loop(cond, body, (-1, len(points) - 1))
    return lax.cond(x < points[high], lambda: low, lambda: high) + 1


def _get_cos_curve_at_x(
    x: float,
    x_data,
    y_data,
) -> float:
    """Get interpolated function value using half-cosine function.

    Args:
        x: Independent value to calculate result at
        x_data: Requested series of independent values
        y_data: Requested series of dependent values

    Returns:
        Interpolated value

    Notes
    -----
    The cosine function was obtained by translating
    and scaling a half cosine function
    (i.e. a cosine function with support $[0, \pi]$),
    such that it intersected the starting point
    $(t_{{1}}, y_{{1}})$ and finishing point $(t_{{2}}, y_{{2}})$
    with a gradient of zero at both of these points.
    This choice of fitting approach ensured that
    the residual transmission scaling function, its derivative
    and its higher order derivatives are continuous.
    """
    idx = binary_search_sum_ge(x, x_data.points) - 1
    offset = x - x_data.points[idx]
    relx = offset / x_data.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return y_data.points[idx] + (rely * y_data.ranges[idx])


class MultiCurve:
    """Abstract class for fitting a curve to a series of data."""

    def get_multicurve(self):
        pass

    def get_description(self):
        pass


class CosineMultiCurve(MultiCurve):
    """Fit a cosine-based curve to a series of data.
    See get_description below for details.

    Args:
        MultiCurve: Abstract parent class
    """

    def get_multicurve(
        self,
        t: float,
        x_data,
        y_data,
    ) -> callable:
        """Construct a half-cosine-based multi-curve.

        Args:
            t: Model time
            x_data: Values of independent variable
            y_data: Values of dependent variable

        Returns:
            Curve fitting function
        """
        # Branch on whether t is in bounds
        bounds_state = sum(t > x_data.bounds)
        branches = [
            lambda _, __, ___: y_data.bounds[0],
            _get_cos_curve_at_x,
            lambda _, __, ___: y_data.bounds[1],
        ]
        return lax.switch(bounds_state, branches, t, x_data, y_data)


from collections import namedtuple

InterpolatorScaleData = namedtuple(
    "InterpolatorScaleData", ["points", "ranges", "bounds"]
)


def get_scale_data(points) -> InterpolatorScaleData:
    """
    Precompute ranges (diffs) and bounds (left and right extrema) for a set of data to be used in
    a scaling function such as that produced by build_sigmoidal_multicurve.  The onus is on the
    caller of this function to ensure they are the length expected by the target callee
    """
    ranges = jnp.diff(points)
    lpoint = points[0]
    rpoint = points[-1]

    # data = {"min": ymin, "max": ymax, "values": points, "ranges": ranges}
    return InterpolatorScaleData(points, ranges, jnp.array([lpoint, rpoint]))
