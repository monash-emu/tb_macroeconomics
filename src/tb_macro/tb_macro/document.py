"""Pull methods text from the code, formatting numeric values on the way out."""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd

import tb_macro.constants as constants
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS

_NOTES_SPLIT = re.compile(r"\nNotes:\s*\n(?:-+\s*\n)?", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


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


def _placeholders() -> dict[str, str]:
    """Map {{name}} keys to formatted values from code.
    Currently includes every variable from constants.py.
    Further mapping can be extended to other files here.
    """
    namespace: dict[str, str] = {}
    for name, value in vars(constants).items():
        if not name.isupper():
            continue
        if isinstance(value, (Path, type, ModuleType, dict)) or callable(value):
            continue
        namespace[name] = md_number(value)

    return namespace


def _interpolate(text: str, namespace: dict[str, str]) -> str:
    missing = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in namespace:
            return namespace[key]
        missing.append(key)
        return match.group(0)

    rendered = _PLACEHOLDER.sub(replace, text)
    if missing:
        names = ", ".join("{{" + key + "}}" for key in missing)
        raise KeyError(f"Unknown documentation placeholders: {names}")
    return rendered


def get_func_notes(function: Callable) -> str:
    """Return the Notes section of a function, with code values interpolated.
    The docstring should include a Google-style Notes section. Placeholders
    such as {{breakdown_rate}} are to be replaced from variables in constants.py.

    Args:
        function: The function containing the notes to render

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
    return _interpolate(notes, _placeholders())


def build_fixed_params_table(params: dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of formatted parameter values."""
    vals = [md_number(val) for val in params.values()]
    return pd.DataFrame({"value": vals}, index=params.keys()).rename_axis("Parameter")


def build_prior_ranges_table(bounds: dict[str, list[float]]) -> pd.DataFrame:
    """Return a DataFrame of formatted prior bounds."""
    vals = [(md_number(low), md_number(high)) for low, high in bounds.values()]
    cols = ["lower", "upper"]
    return pd.DataFrame(vals, index=bounds.keys(), columns=cols).rename_axis("Parameter")
