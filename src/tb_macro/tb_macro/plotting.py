import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from summer3.epi import Stratification, ManagedArray

from tb_macro.constants import AGE_STRATA
from tb_macro.outputs import get_complete_strat_props, get_partial_strat_props

pd.options.plotting.backend = "matplotlib"


def plot_comp_distributions(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    clin_strat: Stratification,
    plot_end_time: float,
    group_popsize,
) -> plt.figure:
    """Plot the distribution of the population by compartment
    or stratification.

    Args:
        results: Results from a single model run
        disease_state: The compartment stratification
        age_strat: The age stratification
        infect_strat: The infectiousness stratification
        clin_strat: The clinical stratification

    Returns:
        The figure
    """
    total_pop = results["compartments"].sum(to_dims="time")
    dstate_props = get_complete_strat_props(results, disease_state)
    age_vals = results["compartments"].sumcats(compartment=age_strat.categories())
    age_props = get_complete_strat_props(results, age_strat)
    inf_props = get_partial_strat_props(results, infect_strat)
    clin_props = get_partial_strat_props(results, clin_strat)

    fig, axes = plt.subplots(2, 3, figsize=[15, 7], sharex=True)
    total_pop.to_pandas_df().plot.area(ax=axes[0, 0], title="total population versus target data", xlim=[1920, plot_end_time - 1], legend=False)
    group_popsize.sum(axis=1).plot(ax=axes[0, 0], linewidth=0.0, color="k", marker="o", markersize=1.0)
    dstate_props.to_pandas_df().clip(lower=0).plot.area(ax=axes[1, 0], title="disease state distribution", ylim=[0.0, 1.0])
    age_vals.to_pandas_df().plot.area(ax=axes[0, 1], title="age group sizes")
    age_props.to_pandas_df().plot.area(ax=axes[0, 2], title="age distribution", ylim=[0.0, 1.0])
    clin_props.to_pandas_df().plot.area(ax=axes[1, 1], title="clinical status distribution", ylim=[0.0, 1.0])
    inf_props.to_pandas_df().plot.area(ax=axes[1, 2], title="infectiousness status distribution", ylim=[0.0, 1.0])
    for ax in axes.ravel():
        ax.legend(loc="upper left")
    plt.close()
    return fig


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
    plt.close()
    return fig


COUNT_TITLES = {
    "prevalence": "prevalent cases",
    "incidence": "incident cases per year",
    "notifications": "notified cases per year",
    "deaths": "deaths per year",
    "latent": "infected population number"
}
RATE_TITLES = {
    "prevalence": "prevalence per 100,000",
    "incidence": "incidence per 100,000 per year",
    "notifications": "notifications per 100,000 per year",
    "deaths": "deaths per 100,000 per year",
    "latent": "percentage with latent infection"
}


def plot_outputs(
    prev: pd.DataFrame, 
    inc: pd.DataFrame, 
    notif: pd.DataFrame,
    notif_target: pd.Series,
    tb_death: pd.DataFrame,
    death_target: pd.Series,
    latent: pd.DataFrame,
    latent_target: pd.Series,
    total_pop: pd.DataFrame,
    plot_start: float,
    end_time: float,
    mode: str,
) -> plt.figure:
    """Plot outputs from multiple model runs.

    Args:
        prev: Prevalence data
        inc: Incidence data
        notif: Notification data
        notif_target: Notification target - a count (i.e. cases per year)
        tb_death: Death data
        death_target: Deaths target - a count (i.e. deaths per year)
        latent: Latent data
        latent_target: Latent target - a percentage
        total_pop: Population size data
        plot_start: Year to plot from
        end_time: End of simulation run
        mode: Whether to plot counts or rates

    Returns:
        The figure
    """
    fig, axes = plt.subplots(3, 2, figsize=[12, 10], sharex=True)
    if mode == "count":
        titles = COUNT_TITLES
        notif_target.plot(ax=axes[1, 0], linewidth=0.0, marker="o", color="k")
        death_target.plot(ax=axes[1, 1], linewidth=0.0, marker="o", color="k")
        data = {
            "prevalence": prev,
            "incidence": inc,
            "notifications": notif,
            "deaths": tb_death,
            "latent": latent,
        }

    elif mode == "rate":
        titles = RATE_TITLES
        latent_target.plot(ax=axes[2, 0], linewidth=0.0, marker="o", color="k")
        data = {
            "prevalence": prev.div(total_pop, axis=0) * 1e5,
            "incidence": inc.div(total_pop, axis=0) * 1e5,
            "notifications": notif.div(total_pop, axis=0) * 1e5,
            "deaths": tb_death.div(total_pop, axis=0) * 1e5,
            "latent": latent.div(total_pop, axis=0) * 1e2,
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'count' or 'rate'.")


    ax_locs = {
        "prevalence": axes[0, 0],
        "incidence": axes[0, 1],
        "notifications": axes[1, 0],
        "deaths": axes[1, 1],
        "latent": axes[2, 0],
    }

    for out in ax_locs:
        data_to_plot = data[out]
        data_to_plot[data_to_plot.index > plot_start].plot(
            ax=ax_locs[out], 
            title=titles[out], 
            legend=False, 
            xlim=[plot_start, end_time - 1],
        )
    axes[2, 1].set_axis_off()

    for ax in axes.ravel():
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    return fig


def plot_age_population_comparison(results, target_pop, age_strat, year):
    modelled = (
        results["compartments"]
        .sumcats(compartment=age_strat.categories())
        .to_pandas_df()
        .loc[year]
        .astype(float)
    )
 
    target = target_pop.loc[year].astype(float)
    target.index = target.index.astype(str)

    df = pd.DataFrame(
        {"age_group": target.index, "Target": target.values, "Modelled": modelled.reindex(target.index).values}
    )
    plot_df = df.melt(id_vars="age_group", var_name="series", value_name="population")

    ax = sns.barplot(data=plot_df, x="age_group", y="population", hue="series", dodge=True)
    ax.set_title(f"population by age in {int(year)}")
    return ax
    