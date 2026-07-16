import pandas as pd
from pathlib import Path
from datetime import datetime, UTC

from summer3.epi import ManagedArray, Stratification


def get_total_pop(
    results: dict,
) -> ManagedArray:
    """Get the total modelled population.

    Args:
        results: Single run results

    Returns:
        Total population over time
    """
    return results["compartments"].sum(to_dims="time")


def get_complete_strat_props(
    results: dict,
    strat: Stratification,
) -> ManagedArray:
    """Get the distribution of the population over
    a stratification that is applied to the whole population.

    Args:
        results: Single run results
        strat: The stratification

    Returns:
        The proportional population distribution
    """
    vals = results["compartments"].sumcats(compartment=strat.categories())
    return vals / get_total_pop(results)


def get_partial_strat_props(
    results: dict,
    strat: Stratification,
) -> ManagedArray:
    """Get the distribution of the population over
    a stratification that is applied to part of the population.

    Args:
        results: Single run results
        strat: The stratification

    Returns:
        The proportional population distribution
    """
    vals = results["compartments"].sumcats(compartment=strat.categories())
    strat_total = vals.sum(to_dims="time")
    return vals / strat_total


def get_share_folder_file_path(
    gdrive_path: str,
):
    """Get path for storing file in GDrive folder for collaborators.

    Args:
        gdrive_path: The local path to GDrive
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    run_folder = f"outputs_{timestamp}"
    gdrive_folder = "Shared drives/EMU Drive/Projects/tb_macro"
    output_path = Path(gdrive_path) / gdrive_folder
    run_path = output_path / run_folder
    run_path.mkdir(exist_ok=True)
    return run_path
