import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from summer3.epi import ManagedArray, Stratification

from tb_macro.constants import AGE_STRATA, PREV_STATES, LATENT_STATES


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
    """Get teh distribution of the population over
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


def get_pop_prev(
    results: dict,
    disease_strat: Stratification,
) -> ManagedArray:
    """Get the prevalence of active disease per 100,000 population.

    Args:
        results: Single run results
        disease_strat: The disease state stratification

    Returns:
        The prevalence
    """
    vals = results["compartments"].sumcats(
        compartment=disease_strat.categories(PREV_STATES)
    )
    total_pop = results["compartments"].sum(to_dims="time")
    prev = vals.sum(to_dims="time")
    return prev / total_pop * 1e5


def get_mort(
    results: dict,
) -> pd.DataFrame:
    """Get the TB-related mortality absolute numbers
    disaggregated into community and treatment-related deaths.

    Args:
        results: Single run results

    Returns:
        The mortality data
    """
    community_mort = results["flows"]["tb_mortality"].sum(to_dims="time")
    rx_mort = results["flows"]["rx_death"].sum(to_dims="time")
    tb_mort = pd.concat(
        [community_mort.to_pandas_df(), rx_mort.to_pandas_df()],
        axis=1,
    ).clip(lower=0.0)
    tb_mort.columns = ["community", "treatment"]
    return tb_mort


def get_latent_percentage(
    results: dict,
    disease_strat: Stratification,
) -> ManagedArray:
    """Get the percentage of the population ever
    previously infected.

    Args:
        results: Single run results

    Returns:
        The latent percentage
    """
    latent_vals = results["compartments"].sumcats(
        compartment=disease_strat.categories(LATENT_STATES)
    )
    latent_prev = latent_vals.sum(to_dims="time")
    total_pop = get_total_pop(results)
    return latent_prev / total_pop * 1e2


def plot_dynamic_mixing_matrix(
    dmm: ManagedArray,
    start: float,
    interval: float,
    n_cols: int,
):
    """Plot dynamic mixing matrices in multipanel figure.

    Args:
        dmm: The dynamic mixing matrix computed value
        start: The first year to plot
        interval: The interval between years
        n_cols: The number of columns for the plots
            (determines the number of panels - will be n_cols x2)
    """
    n_rows = 2
    figsize = [4 * n_cols, 7]
    dmm_xa = dmm.to_xarray_da()
    vmax = dmm_xa.max()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    flat_axes = axes.ravel()
    for a, ax in enumerate(flat_axes):
        year = start + a * interval
        hm = sns.heatmap(
            dmm_xa.sel(time=year),
            cmap="viridis",
            xticklabels=AGE_STRATA,
            yticklabels=AGE_STRATA[::-1],
            ax=ax,
            vmin=0.0,
            vmax=vmax,
            cbar=False,
        )
        ax.set_title(int(year))
    im = hm.collections[0]
    fig.colorbar(im, ax=axes, shrink=0.8)
