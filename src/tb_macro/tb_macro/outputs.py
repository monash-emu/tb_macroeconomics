import matplotlib.pyplot as plt
import seaborn as sns

from summer3.epi import ManagedArray

from tb_macro.constants import AGE_STRATA


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
    