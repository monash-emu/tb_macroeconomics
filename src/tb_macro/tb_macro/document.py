"""Pull methods text from the code, formatting numeric values on the way out."""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Callable

import pandas as pd

from tb_macro.constants import (
    AGE_STRATA,
    END_TIME,
    ISO3,
    MAX_AGE,
    START_TIME,
    YOUNG_END_AGE,
)
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS

_NOTES_SPLIT = re.compile(r"\nNotes:\s*\n(?:-+\s*\n)?", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def md_number(value: Any, sci_exp_threshold: int = 4) -> str:
    """Format a value for Markdown, using LaTeX scientific notation when needed.

    Numeric values stay as Python floats in the codebase. This is the only
    place they become display strings.

    Args:
        value: Value to format
        sci_exp_threshold: Use $a \\times 10^{b}$ when |log10| is at least this

    Returns:
        Markdown/LaTeX string
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, (list, tuple)):
            return ", ".join(md_number(v, sci_exp_threshold) for v in value)
        return str(value)

    x = float(value)
    if x == 0.0:
        return "0"

    exp = math.floor(math.log10(abs(x)))
    if abs(exp) >= sci_exp_threshold:
        mantissa = x / (10**exp)
        return rf"${mantissa:.3g} \times 10^{{{exp}}}$"

    if abs(x - round(x)) < 1e-12 * max(1.0, abs(x)):
        return str(int(round(x)))
    return f"{x:g}"


def get_doc_namespace(
    params: dict[str, Any] | None = None,
    bounds: dict[str, list[float]] | None = None,
) -> dict[str, str]:
    """Build the interpolation mapping from code values.

    Each base parameter becomes `{name}`, and each prior bound becomes
    `{name}_low` / `{name}_up`. Selected constants keep their Python names.
    """
    params = BASE_PARAMS if params is None else params
    bounds = PARAM_BOUNDS if bounds is None else bounds

    namespace = {name: md_number(value) for name, value in params.items()}
    for name, (low, high) in bounds.items():
        namespace[f"{name}_low"] = md_number(low)
        namespace[f"{name}_up"] = md_number(high)

    namespace.update(
        {
            "START_TIME": md_number(START_TIME),
            "END_TIME": md_number(END_TIME),
            "YOUNG_END_AGE": md_number(YOUNG_END_AGE),
            "MAX_AGE": md_number(MAX_AGE),
            "ISO3": ISO3,
            "AGE_STRATA": md_number(AGE_STRATA),
        }
    )
    return namespace


def _interpolate(text: str, namespace: dict[str, str]) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in namespace:
            return namespace[key]
        # Leave one-character braces alone (LaTeX subscripts such as $x_{i}$)
        if len(key) == 1:
            return match.group(0)
        missing.append(key)
        return match.group(0)

    rendered = _PLACEHOLDER.sub(replace, text)
    if missing:
        names = ", ".join(f"{{{key}}}" for key in missing)
        raise KeyError(f"Unknown documentation placeholders: {names}")
    return rendered


def get_func_notes(
    function: Callable,
    namespace: dict[str, str] | None = None,
) -> str:
    """Return the Notes section of a function, with code values interpolated.

    The docstring should include a Google-style Notes section. Placeholders
    such as ``{breakdown_rate}`` are replaced from :func:`get_doc_namespace`.

    Args:
        function: Function whose Notes should be rendered
        namespace: Interpolation mapping; defaults to :func:`get_doc_namespace`

    Returns:
        Markdown text

    Raises:
        ValueError: If the function has no Notes section
        KeyError: If a placeholder is not in the namespace
    """
    docstring = inspect.getdoc(function)
    if not docstring:
        raise ValueError(f"{function.__name__} has no docstring")

    parts = _NOTES_SPLIT.split(docstring, maxsplit=1)
    if len(parts) < 2:
        raise ValueError(f"{function.__name__} has no Notes section")

    notes = parts[1].strip()
    return _interpolate(notes, get_doc_namespace() if namespace is None else namespace)


def build_fixed_params_table(params: dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of formatted parameter values."""
    return pd.DataFrame(
        {"Value": [md_number(value) for value in params.values()]},
        index=list(params),
    ).rename_axis("Parameter")


def build_prior_ranges_table(bounds: dict[str, list[float]]) -> pd.DataFrame:
    """Return a DataFrame of formatted prior bounds."""
    return pd.DataFrame(
        [(md_number(low), md_number(high)) for low, high in bounds.values()],
        index=list(bounds),
        columns=["Lower", "Upper"],
    ).rename_axis("Parameter")
