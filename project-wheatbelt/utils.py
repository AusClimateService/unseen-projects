"""Utility functions for low/high growing season (Apr-Oct) rainfall events."""

import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import scipy
import xarray as xr

from process_gsr_data import home, models


def binom_ci(n, p=0.3):
    """Apply binomial test to determine confidence intervals."""
    ci0, ci1 = xr.apply_ufunc(
        scipy.stats.binom.interval,
        0.95,
        n,
        input_core_dims=[[], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(p=p),
        output_dtypes=["float64", "float64"],
    )
    return ci0, ci1


def plot_aus_map(
    fig,
    ax,
    data,
    title=None,
    outfile=False,
    cbar_kwargs=dict(fraction=0.05, extend="max"),
    **kwargs,
):
    """Plot 2D data on an Australia map with coastlines.

    Parameters
    ----------
    fig :
    ax : matplotlib plot axis
    data : xarray DataArray
        2D data to plot
    title : str, optional
        Title for the plot
    outfile : str, optional
        Filename for the plot
    cb_kwargs : dict, optional
        Additional keyword arguments for colorbar.
    **kwargs : optional
        Additional keyword arguments for pcolormesh

    Returns
    -------
    ax : matplotlib plot axis

    Example
    -------
    import cartopy.crs as ccrs
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax = plot_aus_map()
    """
    if title is not None:
        ax.set_title(title, loc="left")

    cs = ax.pcolormesh(data.lon, data.lat, data, zorder=0, **kwargs)
    fig.colorbar(cs, **cbar_kwargs)

    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, zorder=1)
    ax.coastlines()
    # Format ticks
    xticks = np.arange(115, 155, 5)
    yticks = np.arange(-40, -10, 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        axis="both",
        which="both",
        direction="inout",
        length=7,
        bottom=True,
        top=True,
        left=True,
        right=True,
        zorder=4,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="inout",
        length=3,
    )
    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches="tight", dpi=200)
        plt.show()
    return ax


def combine_figures(files, axes, axis=False):
    """Combine plotted figures of single models into a image."""
    files = sorted(files)
    outfile = str(files[-1]).replace("_NorCPM1", "-combined")

    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i < len(files):
            img = mpl.image.imread(files[i])
            ax.imshow(img)
            ax.axis(axis)
            ax.tick_params(
                axis="both",
                which="both",
                left=False,
                right=False,
                top=False,
                bottom=False,
            )
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=400)
    plt.show()


def combine_all_figures():
    """Combine all plotted figures of single models into a image."""
    path = home / "figures"

    for s in ["LGSR_", "HGSR_"]:
        # Combine as 4x3 grid
        for f in ["transition_duration_histogram_*.png"]:
            files = [f"{path}/{s}{f}".replace("*", str(m)) for m in ["AGCD", *models]]
            _, axes = plt.subplots(3, 4, figsize=[12, 7], layout="compressed")
            combine_figures(files, axes, axis=True)

        # Combine as 3x3 grid
        for f in [
            "duration_histogram_xsamples_*.png",
            "transition_histogram_xsamples_*.png",
            "transition_sample_size_*.png",
        ]:
            files = [f"{path}/{s}{f}".replace("*", str(m)) for m in models]
            _, axes = plt.subplots(3, 3, figsize=[12, 10], layout="compressed")
            combine_figures(files, axes, axis=True)

        # Combine as 2x5 grid
        for f in [
            "duration_histogram_*.png",
            "map_duration_*.png",
            "map_pr_2yr_*.png",
            "map_pr_3yr_*.png",
            "map_transition_probability_1yr_*.png",
            "map_transition_probability_2yr_*.png",
            "map_transition_probability_3yr_*.png",
            "transition_histogram_binned_decile_*.png",
            "transition_histogram_decile_*.png",
            "transition_histogram_tercile_*.png",
            "transition_matrix_*.png",
            "transition_pie_*.png",
        ]:
            files = [f"{path}/{s}{f}".replace("*", str(m)) for m in ["AGCD", *models]]
            _, axes = plt.subplots(5, 2, figsize=[8, 6], layout="compressed")
            combine_figures(files, axes, axis=True)

        # Combine as 2x5 grid (no subplot grid)
        for f in [
            "map_count_2yr_*.png",
            "map_count_3yr_*.png",
            "map_frequency_2yr_*.png",
            "map_frequency_3yr_*.png",
            "map_persistance_probability_*.png",
        ]:
            files = [f"{path}/{s}{f}".replace("*", str(m)) for m in ["AGCD", *models]]
            _, axes = plt.subplots(5, 2, figsize=[8, 6], layout="compressed")
            combine_figures(files, axes, axis=False)
