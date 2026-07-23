from typing import List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC

from summer3.epi import ManagedArray, Stratification

from tb_macro.constants import PREV_STATES, LATENT_STATES, AGE_STRATA


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
            scen_outs.append(
                pd.concat(output, axis=1, keys=sample_labels, names=["sample"])
            )
        full_outs.append(
            pd.concat(scen_outs, axis=1, keys=indicators, names=["indicator"])
        )
    return pd.concat(full_outs, axis=1, keys=range(n_scenarios), names=["scenario"])


def is_age_stratified_output(
    output: pd.DataFrame,
) -> bool:
    """Determine whether single output dataframe is
    stratified by age.

    Args:
        output: The output data

    Returns:
        Age stratification status
    """
    age_labels = {str(a) for a in AGE_STRATA}
    return (
        len(output.columns) == len(age_labels)
        and {str(col) for col in output.columns} == age_labels
    )


def assign_age_groups(
    pops: pd.DataFrame,
    breaks: List[int],
    name: str,
):
    """Classify single year ages into age groups
    and add this information to the population data argument.

    Args:
        pops: Population sizes (or any data) by single years of age
        breaks: The age group breakpoints
        name: The name for the newly created column
    """
    break_ints = [int(a) for a in breaks]
    bins = break_ints + [np.inf]
    pops[name] = pd.cut(pops["Age"], bins=bins, right=False, labels=break_ints)


def build_age_mapping(
    pops: pd.DataFrame,
    m_group_name: str,
    o_group_name: str,
) -> pd.DataFrame:
    """Calculate the fraction of each modelled age group
    that should be assigned to each output age group.

    Args:
        pops: Population sizes by single years of age,
            including mapping of modelled and output age groups
        m_group_name: The column name for the modelled age group mapping
        o_group_name: The column name for the output age group mapping

    Returns:
        The mapping object
    """
    pops = pops.copy()

    # Total population overlapping between the two age groups specified in the modelled and output columns
    overlaps = (
        pops.groupby(["Time", m_group_name, o_group_name])["Pop"].sum().reset_index()
    )

    # Calculate the denominator - the population in the modelled age group
    model_totals = pops.groupby(["Time", m_group_name])["Pop"].sum().reset_index()

    # Assign the denominators to every row of the overlaps object
    mapping = overlaps.merge(
        model_totals, on=["Time", m_group_name], suffixes=("_overlap", "_model")
    )

    # Calculate the fraction of each modelled age group to assign to the output age group
    mapping["fraction"] = mapping["Pop_overlap"] / mapping["Pop_model"]

    # Tidy up
    relevant_cols = ["Time", m_group_name, o_group_name, "fraction"]
    mapping[m_group_name] = mapping[m_group_name].astype(str)
    return mapping[relevant_cols]


def regroup_output(
    output: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Use mapping from modelled to output
    age groups to regroup output.

    Args:
        output: The output with age groups assigned
        mapping: The mapping object from modelled to output age groups

    Returns:
        The regrouped output
    """
    out_long = output.reset_index(names="Time").melt(
        id_vars="Time", var_name="model_agegroup", value_name="value"
    )
    weighted = out_long.merge(mapping, on=["Time", "model_agegroup"])
    weighted["value"] *= weighted["fraction"]
    regrouped = (
        weighted.groupby(["Time", "output_agegroup"])["value"].sum().reset_index()
    )
    return regrouped.pivot(index="Time", columns="output_agegroup", values="value")


def map_regroup_output(
    output: pd.DataFrame,
    single_age_pops: pd.DataFrame,
    out_groups: List[int],
) -> pd.DataFrame:
    """Take an output structured by modelled age
    and restructure according to
    requested output age groups.

    Args:
        output: The output data

    Returns:
        The restructured data
    """
    assign_age_groups(single_age_pops, output.columns, "model_agegroup")
    assign_age_groups(single_age_pops, out_groups, "output_agegroup")
    mapping = build_age_mapping(single_age_pops, "model_agegroup", "output_agegroup")
    return regroup_output(output, mapping)


def regroup_full_outputs(
    outputs: List[Dict[str, List[pd.DataFrame]]],
    regrouped_outputs: List[Dict[str, List[pd.DataFrame]]],
    single_age_pops: pd.DataFrame,
    out_groups: List[int],
) -> List[Dict[str, List[pd.DataFrame]]]:
    """Take the full outputs data structure and apply
    the regrouping process to each internal element
    to get the raw regrouped outputs structure.

    Args:
        outputs: The outputs structured by modelled age groups
        regrouped_outputs: The empty output structure to populate
        single_age_pops: The population data in single year age groups
        out_groups: The requested output age breakpoints

    Returns:
        The regrouped outputs
    """
    for s, scenario_outputs in enumerate(outputs):
        for ind, raw_outputs in scenario_outputs.items():
            for output in raw_outputs:
                if output.columns.name == "age_group":
                    regrouped_output = map_regroup_output(
                        output, single_age_pops, out_groups
                    )
                else:
                    regrouped_output = output
                regrouped_outputs[s][ind].append(regrouped_output)
    return regrouped_outputs
