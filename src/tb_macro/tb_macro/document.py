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
from tb_macro.parameters import PARAM_NAMES


_NOTES_SPLIT = re.compile(r"\nNotes:\s*\n(?:-+\s*\n)?")
_PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")


def md_number(value: Any, sci_exp_threshold: int = 4) -> str:
    """Format a value for Markdown, using LaTeX scientific notation when needed.

    Args:
        value: Value to format
        sci_exp_threshold: Use scientific notation when the base-10 exponent
            is at least this far from zero, so both very large and very
            small magnitudes qualify

    Returns:
        Markdown/LaTeX string
    """

    # If not numeric (Python bools are a subclass of int)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if isinstance(value, (list, tuple)):
            return ", ".join(md_number(v, sci_exp_threshold) for v in value)
        return str(value)

    # Handled separately because log10 is undefined at zero
    x = float(value)
    if x == 0.0:
        return "0"

    # Scientific notation for very large or small positive or negative magnitudes
    exp = math.floor(math.log10(abs(x)))
    if abs(exp) >= sci_exp_threshold:
        mantissa = x / (10**exp)
        return rf"${mantissa:.3g} \times 10^{{{exp}}}$"

    # Round numbers close to whole numbers
    if abs(x - round(x)) < 1e-12 * max(1.0, abs(x)):
        return str(int(round(x)))

    # Otherwise standard formatting for numeric
    return f"{x:g}"


def _get_name_mapping() -> dict[str, str]:
    """Map {{name}} keys to formatted values from code.
    Currently includes every variable from constants.py.
    Further mapping can be extended to other files here.
    """
    namespace = {}
    for name, value in vars(constants).items():

        # Variable names must be in upper case
        if not name.isupper():
            continue

        # Ignore paths, dicts, modules, classes and functions
        if isinstance(value, (Path, type, ModuleType, dict)) or callable(value):
            continue

        # Scalars are formatted for display, lists are comma-joined
        namespace[name] = md_number(value)

    return namespace


def _interpolate_notes_str(text: str, namespace: dict[str, str]) -> str:
    """Substitute every {{name}} in the text with its value from the namespace."""

    # Check all the names first, so one error can report every unknown name
    missing = [key for key in _PLACEHOLDER.findall(text) if key not in namespace]
    if missing:
        names = ", ".join("{{" + key + "}}" for key in missing)
        raise KeyError(f"Unrecognised documentation names: {names}")

    return _PLACEHOLDER.sub(lambda match: namespace[match.group(1)], text)


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

    notes = parts[1].strip() # Text after notes without leading/trailing whitespace
    return _interpolate_notes_str(notes, _get_name_mapping())


def build_fixed_params_table(params: dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of formatted parameter values."""
    vals = [md_number(val) for val in params.values()]
    index = [PARAM_NAMES.get(param, param) for param in params]
    return pd.DataFrame({"value": vals}, index=index).rename_axis("Parameter")


def build_prior_ranges_table(bounds: dict[str, list[float]]) -> pd.DataFrame:
    """Return a DataFrame of formatted prior bounds."""
    vals = [(md_number(low), md_number(high)) for low, high in bounds.values()]
    index = [PARAM_NAMES.get(param, param) for param in bounds]
    cols = ["lower", "upper"]
    return pd.DataFrame(vals, index=index, columns=cols).rename_axis("Parameter")
