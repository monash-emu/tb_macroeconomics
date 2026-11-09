from typing import Dict, List, Optional

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from arviz import InferenceData
from jax import numpy as jnp
from numpyro import distributions as dist

from summer3.epi import Stratification, ManagedArray

from tb_macro.constants import (
    AGE_STRATA,
    INFECT_COMPS,
    INFECTED_STATES,
    ISO3,
    MIXING_TARGET_YEAR,
    YOUNG_END_AGE,
)
from tb_macro.inputs import get_norm_conmat
from tb_macro.mixing import build_s_matrix_single_age_components, canberra_distance
from tb_macro.outputs import get_complete_strat_props, get_partial_strat_props
from tb_macro.parameters import BASE_PARAMS, PARAM_ABBREVS, PARAM_BOUNDS, PARAM_NAMES
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
    "latent": "adult infected population number",
    "pulm_prev": "adult pulmonary bacteriologically-detectable cases",
}
RATE_TITLES = {
    "prevalence": "prevalence per 100,000",
    "incidence": "incidence per 100,000 per year",
    "notifications": "notifications per 100,000 per year",
    "deaths": "deaths per 100,000 per year",
    "latent": "percentage of adults with latent infection",
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


def _sum_df_over_sample_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the innermost age-group level, leaving sample columns."""
    return df.T.groupby(level=0).sum().T


def get_scenario_indicator_totals(
    good_out: pd.DataFrame,
    young_end_age: int = YOUNG_END_AGE,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Sum age-stratified scenario outputs into sample-path totals.

    Args:
        good_out: Collated multi-scenario outputs with column levels
            scenario / indicator / sample / agegroup
        young_end_age: Ages at or above this threshold count as adult

    Returns:
        Mapping from scenario index to indicator DataFrames whose columns
        are posterior samples
    """
    scenarios: Dict[int, Dict[str, pd.DataFrame]] = {}
    for s in good_out.columns.get_level_values("scenario").unique():
        adult_ages = (
            good_out[s]["age_pop"].columns.get_level_values("agegroup").astype(int)
            >= young_end_age
        )
        scenarios[int(s)] = {
            "total_pop": _sum_df_over_sample_level(good_out[s]["age_pop"]),
            "incidence": _sum_df_over_sample_level(good_out[s]["incidence"]),
            "notifications": _sum_df_over_sample_level(good_out[s]["notifications"]),
            "prevalence": _sum_df_over_sample_level(good_out[s]["prevalence"]),
            "deaths": _sum_df_over_sample_level(good_out[s]["deaths"]),
            "adult_pop": _sum_df_over_sample_level(
                good_out[s]["age_pop"].loc[:, adult_ages]
            ),
            "pulm_prev": _sum_df_over_sample_level(
                good_out[s]["pulm_prev"].loc[:, adult_ages]
            ),
            "latent": _sum_df_over_sample_level(
                good_out[s]["latent"].loc[:, adult_ages]
            ),
        }
    return scenarios


def _median_and_ci(
    df: pd.DataFrame,
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Median and equal-tailed credible interval across sample columns."""
    return (
        df.quantile(0.5, axis=1),
        df.quantile(q_low, axis=1),
        df.quantile(q_high, axis=1),
    )


def plot_scenario_credible_intervals(
    scenario_totals: Dict[int, Dict[str, pd.DataFrame]],
    scenario_labels: Dict[int, str],
    notif_target: Optional[pd.Series],
    death_target: Optional[pd.Series],
    latent_target: Optional[pd.Series],
    plot_start: float,
    end_time: float,
    mode: str,
    pulm_prev_target: Optional[pd.Series],
    prev_decline_target: Optional[pd.Series] = None,
    q_low: float = 0.025,
    q_high: float = 0.975,
) -> plt.figure:
    """Plot median trajectories with credible-interval bands by scenario.

    Uses the same six-panel layout as ``plot_outputs``, but summarises the
    posterior sample paths instead of drawing spaghetti lines.

    Args:
        scenario_totals: Output of ``get_scenario_indicator_totals``
        scenario_labels: Display names keyed by scenario index
        notif_target: Notification target counts, or None to omit
        death_target: Death target counts, or None to omit
        latent_target: Adult latent target percentage, or None to omit
        plot_start: Year to plot from
        end_time: Last year shown on the x-axis upper bound
        mode: Whether to plot counts or rates
        pulm_prev_target: Adult pulmonary prevalence target rate, or None
        prev_decline_target: Culture-positive survey points for the decline arrow
        q_low: Lower quantile of the credible interval
        q_high: Upper quantile of the credible interval

    Returns:
        The figure
    """
    fig, axes = plt.subplots(3, 2, figsize=[12, 10], sharex=True)
    ax_locs = {
        "prevalence": axes[0, 0],
        "incidence": axes[0, 1],
        "notifications": axes[1, 0],
        "deaths": axes[1, 1],
        "latent": axes[2, 0],
        "pulm_prev": axes[2, 1],
    }
    def _in_window(target: Optional[pd.Series]) -> Optional[pd.Series]:
        if target is None:
            return None
        clipped = target.loc[(target.index >= plot_start) & (target.index <= end_time)]
        return clipped if not clipped.empty else None

    if mode == "count":
        titles = COUNT_TITLES
        notif_in_window = _in_window(notif_target)
        death_in_window = _in_window(death_target)
        if notif_in_window is not None:
            notif_in_window.plot(ax=axes[1, 0], linewidth=0.0, marker="o", color="k", zorder=3)
        if death_in_window is not None:
            death_in_window.plot(ax=axes[1, 1], linewidth=0.0, marker="o", color="k", zorder=3)
    elif mode == "rate":
        titles = RATE_TITLES
        latent_in_window = _in_window(latent_target)
        pulm_in_window = _in_window(pulm_prev_target)
        if latent_in_window is not None:
            latent_in_window.plot(ax=axes[2, 0], linewidth=0.0, marker="o", color="k", zorder=4)
        if pulm_in_window is not None:
            pulm_in_window.plot(ax=axes[2, 1], linewidth=0.0, marker="o", color="k", zorder=4)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'count' or 'rate'.")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (s, label) in enumerate(scenario_labels.items()):
        totals = scenario_totals[s]
        color = colors[i % len(colors)]
        if mode == "count":
            data = {
                "prevalence": totals["prevalence"],
                "incidence": totals["incidence"],
                "notifications": totals["notifications"],
                "deaths": totals["deaths"],
                "latent": totals["latent"],
                "pulm_prev": totals["pulm_prev"],
            }
        else:
            data = {
                "prevalence": totals["prevalence"].div(totals["total_pop"], axis=0) * 1e5,
                "incidence": totals["incidence"].div(totals["total_pop"], axis=0) * 1e5,
                "notifications": totals["notifications"].div(totals["total_pop"], axis=0) * 1e5,
                "deaths": totals["deaths"].div(totals["total_pop"], axis=0) * 1e5,
                "latent": totals["latent"].div(totals["adult_pop"], axis=0) * 1e2,
                "pulm_prev": totals["pulm_prev"].div(totals["adult_pop"], axis=0) * 1e5,
            }

        for out, ax in ax_locs.items():
            series = data[out].loc[
                (data[out].index >= plot_start) & (data[out].index <= end_time)
            ]
            median, low, high = _median_and_ci(series, q_low=q_low, q_high=q_high)
            ax.fill_between(series.index, low, high, color=color, alpha=0.25, linewidth=0.0)
            median.plot(ax=ax, color=color, label=label)
            ax.set_title(titles[out])

    if (
        mode == "rate"
        and pulm_prev_target is not None
        and prev_decline_target is not None
        and _in_window(pulm_prev_target) is not None
        and _in_window(prev_decline_target) is not None
    ):
        pulm_ax = ax_locs["pulm_prev"]
        add_pulm_prev_decline_arrow(pulm_ax, pulm_prev_target, prev_decline_target)
        pulm_ax.relim()
        pulm_ax.autoscale_view(scalex=False, scaley=True)

    axes[0, 0].legend(loc="upper right")
    for ax in axes.ravel():
        ax.set_xlim(plot_start, end_time)
        ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    plt.close()
    return fig


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
        latent: Adult latent (ever-infected) counts
        latent_target: Adult latent target - a percentage
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
            "latent": latent.div(adult_pop, axis=0) * 1e2,
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
    the likelihood: notifications, TB deaths, adult latent infection, adult
    pulmonary prevalence (with the survey decline implied by the 2017
    target), and the highly infectious share of that prevalence. The final
    panel summarises modelled / target at the likelihood comparison points.

    Mixing is plotted separately by ``plot_mixing_target_comparison``.
    """
    notif = _managed_to_annual(results["flows"]["detection"])
    deaths = _managed_to_annual(results["flows"]["tb_mortality"]) + _managed_to_annual(
        results["flows"]["rx_death"]
    )
    high_inf, pulm_prev, adult_pop = _adult_pulm_prev_series(
        results, disease_state, age_strat, infect_strat
    )
    adult_ages = age_strat[[str(a) for a in AGE_STRATA if a >= YOUNG_END_AGE]]
    latent = (
        _managed_to_annual(
            results["compartments"].query(
                compartment=(disease_state[INFECTED_STATES], adult_ages)
            )
        )
        / adult_pop
        * 100.0
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
        "percentage of adults with latent infection",
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


_INFECT_SOURCE_LABELS = {
    "mtb_naive": "never-infected",
    "contained": "contained",
    "cleared": "cleared",
    "recovered": "recovered",
}


def plot_infection_source_props(
    results: dict,
    start: float,
    end: float,
) -> plt.figure:
    """Stacked area of new infection proportions by source compartment.

    Args:
        results: Results from a single model run
        start: First year to plot
        end: Last year to plot

    Returns:
        The figure
    """
    total_pop = _managed_to_annual(results["compartments"])
    rates = {}
    for comp in INFECT_COMPS:
        infections = _managed_to_annual(results["flows"][f"infect_{comp}"])
        rates[_INFECT_SOURCE_LABELS[comp]] = infections / total_pop * 1e5
    df = pd.DataFrame(rates).loc[start:end].clip(lower=0.0)
    prop_df = df.div(df.sum(axis=1), axis=0).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    prop_df.plot.area(ax=ax)
    fig.suptitle("infection source by compartment")
    ax.set_xlim(start, end)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("proportion")
    ax.legend(title="source", loc="upper left")
    fig.tight_layout()
    plt.close()
    return fig


def plot_single_age_s_matrix_components(
    fert_padded: pd.DataFrame,
    params: dict,
    year: float = MIXING_TARGET_YEAR,
) -> plt.figure:
    """Plot the single-age S-matrix and its three mixing components.

    Args:
        fert_padded: Fertility rates padded across single years of age
        params: Parameter set containing the mixing scalars
        year: Calendar year at which to evaluate the kernel

    Returns:
        The figure
    """
    background, assortative, parent_child, overall = build_s_matrix_single_age_components(
        jnp.array(fert_padded),
        jnp.array(fert_padded.index[[0, -1]]),
        year,
        params["bg_mixing"],
        params["a_spread"],
        params["pc_strength"],
    )
    panels = (
        (assortative, "assortative"),
        (background, "background"),
        (parent_child, "parent-child"),
        (overall, "overall"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, (matrix, title) in zip(axes.ravel(), panels):
        data = np.asarray(matrix)
        sns.heatmap(data, cmap="viridis", xticklabels=10, yticklabels=10, ax=ax, vmin=0.0, square=True)
        ax.set_title(title)
        ax.set_xlabel("age")
        ax.set_ylabel("age")
    fig.suptitle(f"single-age mixing kernel, {int(year)}")
    plt.close()
    return fig


def plot_mean_contacts_by_age(
    results: dict,
    year: float = MIXING_TARGET_YEAR,
) -> plt.figure:
    """Compare modelled and synthetic mean contacts by age group.

    Both matrices are spectral-radius normalised, as in the likelihood.
    Mean contacts are the row sums of the mixing matrix.

    Args:
        results: Results from a single model run
        year: Calendar year of the modelled mixing matrix

    Returns:
        The figure
    """
    modelled = np.asarray(
        results["computed_values"]["dynamic_mm"].to_xarray_da().sel(time=year)
    )
    target = np.asarray(get_norm_conmat(ISO3))
    df = pd.DataFrame(
        {
            "age_group": [str(a) for a in AGE_STRATA],
            "modelled": modelled.sum(axis=1),
            "target": target.sum(axis=1),
        }
    )
    plot_df = df.melt(id_vars="age_group", var_name="series", value_name="mean_contacts")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="age_group", y="mean_contacts", hue="series", ax=ax)
    ax.set_xlabel("age group")
    ax.set_ylabel("mean contacts")
    ax.set_title(f"mean contacts by age, {int(year)}")
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    plt.close()
    return fig


def _current_priors() -> Dict[str, dist.Distribution]:
    """Uniform priors from the current PARAM_BOUNDS."""
    return {k: dist.Uniform(v[0], v[1]) for k, v in PARAM_BOUNDS.items()}


def _prior_density(distri: dist.Distribution, x_vals: np.ndarray) -> np.ndarray:
    """Evaluate a numpyro distribution density, including batched priors."""
    x_eval = jnp.asarray(x_vals)
    if len(distri.batch_shape) == 0:
        logp = distri.log_prob(x_eval)
    else:
        logp = distri.log_prob(x_eval[:, None])[:, 0]
    return np.exp(np.asarray(logp))


def plot_prior_post(
    idata: InferenceData,
    var_names: List[str],
    n_cols: int = 3,
) -> plt.Figure:
    """Compare posterior densities to the current-repository priors.

    Posterior panels use ``az.plot_density``. Priors are overlaid as a grey
    fill, scaled to the posterior panel height as in the previous project.

    Args:
        idata: Calibration inference data
        var_names: Posterior variables to plot
        n_cols: Number of subplot columns

    Returns:
        The figure
    """
    n_rows = int(np.ceil(len(var_names) / n_cols))
    priors = _current_priors()
    axes = az.plot_density(
        idata, var_names=var_names, shade=0.3, grid=[n_rows, n_cols], figsize=[10, 15]
    )
    for ax in axes.ravel():
        param = ax.title.get_text().split("\n")[0]
        if param not in priors:
            continue
        ax.set_title((PARAM_NAMES | PARAM_ABBREVS)[param])
        distri = priors[param]
        ax_low, ax_high = ax.get_xlim()
        low = float(getattr(distri, "low", ax_low))
        high = float(getattr(distri, "high", ax_high))
        pad = 0.05 * (high - low)
        x_min = low - pad
        x_max = high + pad
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = _prior_density(distri, x_vals)
        y_peak = float(np.nanmax(y_vals))
        y_vals = y_vals * (ax.get_ylim()[1] / y_peak)
        ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
        ax.set_xlim(x_min, x_max)
    fig = ax.figure
    fig.tight_layout()
    plt.close()
    return fig
