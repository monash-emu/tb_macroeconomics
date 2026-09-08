import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jax import numpy as jnp

from summer3.epi import Stratification, ManagedArray

from tb_macro.constants import (
    AGE_STRATA,
    INFECTED_STATES,
    ISO3,
    MIXING_TARGET_YEAR,
    YOUNG_END_AGE,
)
from tb_macro.inputs import get_norm_conmat
from tb_macro.mixing import canberra_distance
from tb_macro.outputs import get_complete_strat_props, get_partial_strat_props
from tb_macro.parameters import BASE_PARAMS
from tb_macro.targets import (
    INF_PREV_TARGET,
    LATENT_TARGET,
    NOTIF_TARGET,
    PREV_DECLINE_TARGET,
    PULM_PREV_TARGET,
)
from tb_macro.utils import annual_to_midyear, interp_annual_to_times

pd.options.plotting.backend = "matplotlib"


def plot_comp_distributions(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    clin_strat: Stratification,
    plot_start_time: float,
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
    total_pop = results["compartments"].sum(to_dims="time").to_pandas_df().loc[plot_start_time: plot_end_time]
    total_pop_target = group_popsize.sum(axis=1).loc[plot_start_time: plot_end_time]
    dstate_props = get_complete_strat_props(results, disease_state).to_pandas_df().loc[plot_start_time: plot_end_time]
    age_vals = results["compartments"].sumcats(compartment=age_strat.categories()).to_pandas_df().loc[plot_start_time: plot_end_time]
    age_props = get_complete_strat_props(results, age_strat).to_pandas_df().loc[plot_start_time: plot_end_time]
    inf_props = get_partial_strat_props(results, infect_strat).to_pandas_df().loc[plot_start_time: plot_end_time]
    clin_props = get_partial_strat_props(results, clin_strat).to_pandas_df().loc[plot_start_time: plot_end_time]

    fig, axes = plt.subplots(2, 3, figsize=[15, 7], sharex=True)
    modelled_pop = total_pop.squeeze()
    modelled_pop = modelled_pop.where(np.isfinite(modelled_pop))
    total_pop_target.plot(ax=axes[0, 0], linewidth=0.0, color="k", marker="o", markersize=2.0, label="target")
    modelled_pop.plot.area(ax=axes[0, 0], title="total population versus target data", label="modelled")
    dstate_props.clip(lower=0).plot.area(ax=axes[1, 0], title="disease state distribution", ylim=[0.0, 1.0])
    age_vals.clip(lower=0).plot.area(ax=axes[0, 1], title="age group sizes")
    age_props.clip(lower=0).plot.area(ax=axes[0, 2], title="age distribution", ylim=[0.0, 1.0])
    clin_props.clip(lower=0).plot.area(ax=axes[1, 1], title="clinical status distribution", ylim=[0.0, 1.0])
    inf_props.clip(lower=0).plot.area(ax=axes[1, 2], title="infectiousness status distribution", ylim=[0.0, 1.0])
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
    "latent": "infected population number",
    "pulm_prev": "adult pulmonary bacteriologically-detectable cases",
}
RATE_TITLES = {
    "prevalence": "prevalence per 100,000",
    "incidence": "incidence per 100,000 per year",
    "notifications": "notifications per 100,000 per year",
    "deaths": "deaths per 100,000 per year",
    "latent": "percentage with latent infection",
    "pulm_prev": "adult pulmonary bacteriologically-detectable prevalence per 100,000",
}


def add_pulm_prev_decline_arrow(
    ax: plt.Axes,
    pulm_prev_target: pd.Series,
    prev_decline_target: pd.Series,
) -> None:
    """Draw the implied 2007 prevalence from the 2017 target and relative decline.

    The arrow starts at the 2017 pulmonary prevalence target and extends back
    and up at the survey decline rate, so the point is
    (base year, 2017 target * 2007 survey / 2017 survey).
    """
    later_year = float(pulm_prev_target.index[0])
    later_prev = float(pulm_prev_target.iloc[0])
    decline = prev_decline_target.sort_index()
    base_year = float(decline.index[0])
    implied_base_prev = later_prev * float(decline.iloc[0]) / float(decline.iloc[1])
    ax.scatter(
        [base_year],
        [implied_base_prev],
        marker="o",
        facecolors="none",
        edgecolors="k",
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(base_year, implied_base_prev),
        xytext=(later_year, later_prev),
        arrowprops={
            "arrowstyle": "->",
            "color": "k",
            "lw": 1.2,
            "shrinkA": 4,
            "shrinkB": 4,
        },
        zorder=5,
    )


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
    pulm_prev: pd.DataFrame,
    pulm_prev_target: pd.Series,
    adult_pop: pd.DataFrame,
    prev_decline_target: pd.Series = None,
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
        pulm_prev: Adult pulmonary bacteriologically-detectable prevalence counts
        pulm_prev_target: Adult pulmonary bacteriologically-detectable prevalence
            target - a rate (i.e. per 100,000)
        adult_pop: Adult population size data
        prev_decline_target: Culture-positive survey points used only to show
            the implied relative decline on the pulmonary prevalence panel

    Returns:
        The figure
    """
    fig, axes = plt.subplots(3, 2, figsize=[12, 10], sharex=True)
    if mode == "count":
        titles = COUNT_TITLES
        notif_target.plot(ax=axes[1, 0], linewidth=0.0, marker="o", color="k", zorder=3)
        death_target.plot(ax=axes[1, 1], linewidth=0.0, marker="o", color="k", zorder=3)
        data = {
            "prevalence": prev,
            "incidence": inc,
            "notifications": notif,
            "deaths": tb_death,
            "latent": latent,
            "pulm_prev": pulm_prev,
        }

    elif mode == "rate":
        titles = RATE_TITLES
        latent_target.plot(ax=axes[2, 0], linewidth=0.0, marker="o", color="k", zorder=4)
        pulm_prev_target.plot(ax=axes[2, 1], linewidth=0.0, marker="o", color="k", zorder=4)
        data = {
            "prevalence": prev.div(total_pop, axis=0) * 1e5,
            "incidence": inc.div(total_pop, axis=0) * 1e5,
            "notifications": notif.div(total_pop, axis=0) * 1e5,
            "deaths": tb_death.div(total_pop, axis=0) * 1e5,
            "latent": latent.div(total_pop, axis=0) * 1e2,
            "pulm_prev": pulm_prev.div(adult_pop, axis=0) * 1e5,
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'count' or 'rate'.")

    ax_locs = {
        "prevalence": axes[0, 0],
        "incidence": axes[0, 1],
        "notifications": axes[1, 0],
        "deaths": axes[1, 1],
        "latent": axes[2, 0],
        "pulm_prev": axes[2, 1],
    }

    for out in ax_locs:
        data_to_plot = data[out]
        data_to_plot[data_to_plot.index > plot_start].plot(
            ax=ax_locs[out],
            title=titles[out],
            legend=False,
            xlim=[plot_start, end_time - 1],
        )

    if (
        mode == "rate"
        and pulm_prev_target is not None
        and prev_decline_target is not None
    ):
        pulm_ax = ax_locs["pulm_prev"]
        add_pulm_prev_decline_arrow(pulm_ax, pulm_prev_target, prev_decline_target)
        pulm_ax.relim()
        pulm_ax.autoscale_view(scalex=False, scaley=True)

    for ax in axes.ravel():
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    plt.close()
    return fig


def plot_age_population_comparison(results, target_pop, age_strat, years):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()

    for ax, year in zip(axes, years):
        modelled = results["compartments"].sumcats(compartment=age_strat.categories()).to_pandas_df().loc[year].astype(float)

        target = target_pop.loc[year].astype(float)
        target.index = target.index.astype(str)

        df = pd.DataFrame(
            {
                "age_group": target.index,
                "target": target.values,
                "modelled": modelled.reindex(target.index).values,
            }
        )
        plot_df = df.melt(id_vars="age_group", var_name="series", value_name="population")

        sns.barplot(data=plot_df, x="age_group", y="population", hue="series", ax=ax)
        ax.set_title(f"population by age in year {int(year)}")

    fig.tight_layout()
    plt.close()
    return fig


def _managed_to_annual(managed) -> pd.Series:
    """Collapse a managed output onto an annual float-indexed series."""
    series = managed.sum(to_dims="time").to_pandas_df().squeeze()
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = series.astype(float)
    series.index = series.index.astype(float)
    return series


def _midyear_window(series: pd.Series, start: float, end: float) -> pd.Series:
    return annual_to_midyear(series).loc[start:end]


def _geom_mean_ratio(modelled, target) -> float:
    modelled = np.maximum(np.asarray(modelled, dtype=float), 1e-32)
    target = np.maximum(np.asarray(target, dtype=float), 1e-32)
    return float(np.exp(np.mean(np.log(modelled / target))))


def _adult_pulm_prev_series(results, disease_state, age_strat, infect_strat):
    """Adult pulmonary prevalence quantities on the annual solver grid.

    Matches the calibration definition: high-infectious adults, a fraction of
    low-infectious adults, and adults on treatment, over the adult population.
    """
    adult_ages = age_strat[[str(a) for a in AGE_STRATA if a >= YOUNG_END_AGE]]
    high_inf = _managed_to_annual(
        results["compartments"].query(compartment=(infect_strat["high"], adult_ages))
    )
    low_inf = _managed_to_annual(
        results["compartments"].query(compartment=(infect_strat["low"], adult_ages))
    )
    on_rx = _managed_to_annual(
        results["compartments"].query(
            compartment=(disease_state["treatment"], adult_ages)
        )
    )
    adult_pop = _managed_to_annual(
        results["compartments"].query(compartment=adult_ages)
    )
    pulm_prev = high_inf + low_inf * BASE_PARAMS["prop_lowinf_bactpos"] + on_rx
    return high_inf, pulm_prev, adult_pop


def _plot_modelled_and_target(ax, modelled, target, title, start, end) -> None:
    modelled.loc[start:end].plot(ax=ax, label="modelled")
    target.plot(ax=ax, linewidth=0.0, marker="o", color="k", label="target", zorder=4)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(start, end)
    ax.legend()
    ax.set_title(title)


def plot_single_run_comparison(
    results,
    disease_state,
    age_strat,
    infect_strat,
    who_mort,
    start,
    end,
):
    """Compare a single run against the epidemiological calibration targets.

    The time-series panels overlay modelled output on the quantities used in
    the likelihood: notifications, TB deaths, latent infection, adult
    pulmonary prevalence (with the survey decline implied by the 2017
    target), and the highly infectious share of that prevalence. The final
    panel summarises modelled / target at the likelihood comparison points.

    Mixing is plotted separately by ``plot_mixing_target_comparison``.
    """
    notif = _managed_to_annual(results["flows"]["detection"])
    deaths = _managed_to_annual(results["flows"]["tb_mortality"]) + _managed_to_annual(
        results["flows"]["rx_death"]
    )
    total_pop = _managed_to_annual(results["compartments"])
    latent = (
        _managed_to_annual(
            results["compartments"].query(compartment=disease_state[INFECTED_STATES])
        )
        / total_pop
        * 100.0
    )
    high_inf, pulm_prev, adult_pop = _adult_pulm_prev_series(
        results, disease_state, age_strat, infect_strat
    )
    pulm_prev_rate = pulm_prev / adult_pop * 1e5
    inf_prop = high_inf / pulm_prev * 100.0

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    _plot_modelled_and_target(
        axes[0, 0],
        _midyear_window(notif, start, end),
        NOTIF_TARGET,
        "notifications per year",
        start,
        end,
    )
    _plot_modelled_and_target(
        axes[0, 1],
        _midyear_window(deaths, start, end),
        who_mort,
        "TB deaths per year",
        start,
        end,
    )
    _plot_modelled_and_target(
        axes[1, 0],
        _midyear_window(latent, start, end),
        LATENT_TARGET,
        "percentage with latent infection",
        start,
        end,
    )
    axes[1, 0].set_ylim(0.0, 100.0)
    pulm_ax = axes[1, 1]
    _plot_modelled_and_target(
        pulm_ax,
        _midyear_window(pulm_prev_rate, start, end),
        PULM_PREV_TARGET,
        "adult pulmonary prevalence per 100,000",
        start,
        end,
    )
    add_pulm_prev_decline_arrow(pulm_ax, PULM_PREV_TARGET, PREV_DECLINE_TARGET)
    pulm_ax.relim()
    pulm_ax.autoscale_view(scalex=False, scaley=True)
    pulm_ax.set_ylim(bottom=0.0)
    _plot_modelled_and_target(
        axes[2, 0],
        _midyear_window(inf_prop, start, end),
        INF_PREV_TARGET * 100.0,
        "high-infectious share of adult pulmonary prevalence (%)",
        start,
        end,
    )

    decline_target = PREV_DECLINE_TARGET.sort_index()
    modelled_decline = interp_annual_to_times(pulm_prev_rate, decline_target.index)
    fit_ratios = pd.Series(
        {
            "notifications": _geom_mean_ratio(
                interp_annual_to_times(notif, NOTIF_TARGET.index),
                NOTIF_TARGET,
            ),
            "deaths": _geom_mean_ratio(
                interp_annual_to_times(deaths, who_mort.index),
                who_mort,
            ),
            "latent": _geom_mean_ratio(
                interp_annual_to_times(latent, LATENT_TARGET.index),
                LATENT_TARGET,
            ),
            "pulm. prevalence": _geom_mean_ratio(
                interp_annual_to_times(pulm_prev_rate, PULM_PREV_TARGET.index),
                PULM_PREV_TARGET,
            ),
            "high-infectious share": _geom_mean_ratio(
                interp_annual_to_times(inf_prop, INF_PREV_TARGET.index),
                INF_PREV_TARGET * 100.0,
            ),
            "prevalence decline": _geom_mean_ratio(
                modelled_decline.iloc[1] / modelled_decline.iloc[0],
                decline_target.iloc[1] / decline_target.iloc[0],
            ),
        }
    )
    fit_ax = axes[2, 1]
    sns.barplot(
        x=fit_ratios.values,
        y=fit_ratios.index,
        ax=fit_ax,
        color="C0",
        orient="h",
    )
    fit_ax.axvline(1.0, color="k", linewidth=1.0)
    fit_ax.set_xlabel("modelled / target")
    fit_ax.set_ylabel("")
    fit_ax.set_title("fit at calibration targets")

    fig.tight_layout()
    plt.close()
    return fig


def plot_mixing_target_comparison(
    results,
    year: float = MIXING_TARGET_YEAR,
):
    """Compare the modelled mixing matrix to the synthetic contact matrix.

    Both matrices are spectral-radius normalised, as in the likelihood.
    """
    modelled = np.asarray(
        results["computed_values"]["dynamic_mm"].to_xarray_da().sel(time=year)
    )
    target = np.asarray(get_norm_conmat(ISO3))
    distance = float(canberra_distance(jnp.asarray(modelled), jnp.asarray(target)))
    diff = modelled - target
    vmax = max(float(np.max(modelled)), float(np.max(target)))
    dmax = float(np.max(np.abs(diff)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = (
        (axes[0], modelled, "viridis", 0.0, vmax, f"modelled, {int(year)}"),
        (axes[1], target, "viridis", 0.0, vmax, "synthetic contact matrix"),
        (axes[2], diff, "coolwarm", -dmax, dmax, "modelled minus target"),
    )
    for ax, data, cmap, vmin, panel_vmax, title in panels:
        sns.heatmap(
            data,
            cmap=cmap,
            xticklabels=AGE_STRATA,
            yticklabels=AGE_STRATA,
            ax=ax,
            vmin=vmin,
            vmax=panel_vmax,
            square=True,
        )
        ax.set_title(title)
        ax.set_xlabel("age group")
        ax.set_ylabel("age group")
    fig.suptitle(f"mixing target comparison, Canberra distance {distance:.2f}")
    plt.close()
    return fig
