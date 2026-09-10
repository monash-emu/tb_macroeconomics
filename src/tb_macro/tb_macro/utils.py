from jax import numpy as jnp, lax
import jax
from collections import namedtuple
from datetime import datetime, UTC
import logging
import subprocess
import sys
from pathlib import Path
import pandas as pd

from tb_macro.constants import BASE_PATH, START_TIME, CALENDAR_YEAR_MIDPOINT

InterpolatorScaleData = namedtuple(
    "InterpolatorScaleData", ["points", "ranges", "bounds"]
)


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
    x_data: InterpolatorScaleData,
    y_data: InterpolatorScaleData,
) -> float:
    """Get interpolated function value using half-cosine function.

    Args:
        x: Independent value to calculate result at
        x_data: Requested series of independent values
        y_data: Requested series of dependent values

    Returns:
        Interpolated value
    """
    idx = binary_search_sum_ge(x, x_data.points) - 1
    offset = x - x_data.points[idx]
    relx = offset / x_data.ranges[idx]
    rely = 0.5 + 0.5 * -jnp.cos(relx * jnp.pi)
    return y_data.points[idx] + (rely * y_data.ranges[idx])


def get_cos_multicurve(
    t: float,
    x_data: InterpolatorScaleData,
    y_data: InterpolatorScaleData,
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


def get_scale_data(points) -> InterpolatorScaleData:
    """
    Precompute ranges (diffs) and bounds (left and right extrema) for a set of data to be used in
    a scaling function such as that produced by a piecewise multicurve function. The onus is on the
    caller of this function to ensure they are the length expected by the target callee
    """
    ranges = jnp.diff(points)
    lpoint = points[0]
    rpoint = points[-1]

    # data = {"min": ymin, "max": ymax, "values": points, "ranges": ranges}
    return InterpolatorScaleData(points, ranges, jnp.array([lpoint, rpoint]))


def get_logger(log_file: Path):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    root_logger.setLevel(logging.INFO)

    return root_logger


def get_git_provenance(repo: Path | None = None) -> dict[str, str]:
    """Return the current git commit, branch, and dirty flag.

    Args:
        repo: Repository root. Defaults to the project base path.

    Returns:
        Provenance fields as strings, or 'unknown' if git is unavailable.
    """
    repo = Path(repo) if repo is not None else BASE_PATH

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = "unknown" if status.returncode != 0 else str(bool(status.stdout.strip())).lower()
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }


def write_run_log(path: Path, extra: dict | None = None) -> None:
    """Write a small provenance log next to a saved run.

    Args:
        path: Destination log path
        extra: Optional extra fields (output filenames, scenario params, etc.)
    """
    lines = [f"written: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%MZ')}"]
    for key, value in get_git_provenance().items():
        lines.append(f"{key}: {value}")
    if extra:
        for key, value in extra.items():
            lines.append(f"{key}: {value}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def lerp_annual_output(
    y, 
    query_times, 
    start_time: float = START_TIME,
):
    """Linearly interpolate annual solver output to calendar times.

    Args:
        y: Array whose first axis is annual steps from start_time
        query_times: Calendar times to evaluate, e.g. mid-year points
        start_time: Model start year, matching the solver time grid

    Returns:
        Interpolated values at query_times
    """
    y = jnp.squeeze(jnp.asarray(y))
    years = start_time + jnp.arange(y.shape[0])
    return jnp.interp(jnp.asarray(query_times, dtype=y.dtype), years, y)


def interp_annual_to_times(
    output_df: pd.DataFrame | pd.Series, times,
) -> pd.DataFrame | pd.Series:
    """Linearly interpolate an annually indexed series to query times.

    Args:
        output_df: Solver output indexed by calendar year
        times: Calendar times to evaluate

    Returns:
        The interpolated values at times
    """
    times = pd.Index(times, dtype=float)
    base = output_df.copy()
    base.index = base.index.astype(float)
    out = base.reindex(base.index.union(times)).sort_index().interpolate(method="index")
    return out.reindex(times)


def annual_to_midyear(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Interpolate annual solver output onto mid-year points.
    The last annual point has no following year, so the mid-year
    series is one step shorter than the solver grid.
    """
    years = df.index.astype(float)
    mid_times = years[:-1] + CALENDAR_YEAR_MIDPOINT
    return interp_annual_to_times(df, mid_times)


def add_midyear_points(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Keep 1 January solver values and insert interpolated mid-year points.

    The last annual point has no following year, so no mid-year is added after it.
    """
    years = df.index.astype(float)
    return interp_annual_to_times(df, years.union(years[:-1] + CALENDAR_YEAR_MIDPOINT))
