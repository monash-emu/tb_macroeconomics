from typing import List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC

from summer3.epi import ManagedArray, Stratification

from tb_macro.constants import PREV_STATES, LATENT_STATES


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
    return vals / results["compartments"].sum(to_dims="time")


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


# Functions for extracting outputs from results with common signatures
def get_age_inc(results, age_strat, disease_state):
    return results["flows"]["progression"].sumcats(source=age_strat.categories())


def get_age_prev(results, age_strat, disease_state):
    prev_states = results["compartments"].query(compartment=disease_state[PREV_STATES])
    return prev_states.sumcats(compartment=age_strat.categories())


def get_age_latent(results, age_strat, disease_state):
    latent_states = results["compartments"].query(
        compartment=disease_state[LATENT_STATES]
    )
    return latent_states.sumcats(compartment=age_strat.categories())


def get_age_notifs(results, age_strat, disease_state):
    return results["flows"]["detection"].sumcats(source=age_strat.categories())


def get_age_deaths(results, age_strat, disease_state):
    community_death_age = results["flows"]["tb_mortality"].sumcats(
        source=age_strat.categories()
    )
    rx_death_age = results["flows"]["rx_death"].sumcats(source=age_strat.categories())
    return community_death_age + rx_death_age


def get_total_pop(results, age_strat, disease_state):
    return results["compartments"].sum(to_dims="time")


def get_posterior_samples(idata, n_samples):
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    idxs = np.random.choice(posterior.sizes["sample"], size=n_samples, replace=False)
    return posterior.isel(sample=idxs)


def collate_output_table(
    outputs: List[Dict[str, List[pd.DataFrame]]],
    sample_labels: List[str],
) -> pd.DataFrame:
    """Collate the raw outputs that are structured as
    list with elements representing scenarios
        dict with keys representing indicators
            list with elements representing samples
                dataframe with columns representing age groups
    into one multi-indexed dataframe.

    Args:
        outputs: The outputs in raw form
        sample_labels: The names of the samples (chain and draw numbers linked together)

    Returns:
        The full output dataframe
    """
    n_scenarios = len(outputs)
    indicators = list(outputs[0])
    full_outs = []
    for s in range(n_scenarios):
        scen_outs = []
        for out in indicators:
            output = outputs[s][out]
            scen_outs.append(pd.concat(output, axis=1, keys=sample_labels, names=["sample"]))
        full_outs.append(pd.concat(scen_outs, axis=1, keys=indicators, names=["indicator"]))
    return pd.concat(full_outs, axis=1, keys=range(n_scenarios), names=["scenario"])
