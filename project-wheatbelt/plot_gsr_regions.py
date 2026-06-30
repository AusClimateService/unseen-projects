"""Plots of low/high growing season rainfall (GSR) in the WA and SA regions."""

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator
import numpy as np
import pymannkendall
import string
import xarray as xr

from unseen.stability import plot_dist_by_lead, plot_dist_by_time
from unseen.time_utils import str_to_cftime

from process_gsr_data import (
    dataset_names,
    home,
    fig_dir,
    models,
    regions,
    gsr_data_regions,
)
from gsr_events import (
    Events,
    binom_ci,
    get_gsr_events,
    get_event_duration_counts,
    transition_probability,
    transition_time,
    downsampled_transition_probability,
)

plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300

letters = list(string.ascii_lowercase)  # Letters for subplot titles


def transition_probability_cmap(n_intervals=24):
    """Discrete colormap used for transition probabilities.

    Parameters
    ----------
    n_intervals : int
        Number of colormap discrete intervals in the colormap (ideally 12 or 24)

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        Colormap instance.
    norm : matplotlib.colors.BoundaryNorm
        Normalization instance.
    cbar_ticks : array-like
        Colorbar ticks (%) increments of 1/n_intervals to 0.9.
    """

    # Format discrete colormap (purple/orange centred on 1/3)
    # Crop off lower colours so white is ~1/3
    cmap = cmocean.tools.crop_by_percent(
        plt.cm.PuOr, 55 - (100 / 3), which="min", N=None
    )
    # Use 1 / discrete intervals for ticks and cbar
    w = 1 / n_intervals
    cbar_ticks = np.arange(0, 0.9, w) * 100
    levels = np.arange(w / 2, 0.9, w) * 100  # Edges between colours
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")
    return cmap, norm, cbar_ticks


def quantile_cmap(return_colors=False):
    """Discrete colormap used for dry/average/wet quantiles.

    Parameters
    ----------
    return_colors : bool
        If True, instead return array of 3 colours for quantiles.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        Colormap instance
    norm : matplotlib.colors.BoundaryNorm
        Normalization instance
    """

    if return_colors:
        #  Return array of colours for dry/average/wet quantiles
        # rescale = (bins[:-1] - np.min(bins[:-1])) / np.ptp(bins[:-1])
        colors = plt.cm.BrBG(np.array([0.08, 0.42, 0.82]))
        return colors

    # Create colour map of three colours (brown, cream, green)
    cmap = plt.cm.BrBG
    cmap = cmocean.tools.crop_by_percent(cmap, 20, which="min", N=None)
    cmap = cmocean.tools.crop_by_percent(cmap, 20, which="max", N=None)

    cmap = mpl.colors.ListedColormap(colors)
    norm = BoundaryNorm(np.array([0, 1.5, 2.5, 3]), ncolors=cmap.N)
    return cmap, norm


def plot_duration_histogram(dv, event, model, time_dim="time", quantile_var="tercile"):
    """Histogram of event durations (i.e., number of consecutive years).

    Parameters
    ----------
    dv : xarray.Dataset
        Dataset containing pr and tercile data for regions.
    event : Events instance
        Event properties.
    model : str
        Name of the model to show in the title.
    time_dim : str, default "time"
        Name of the core time dimension (in which to look for events).
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.

    Notes
    -----
    - Histogram of the maximum duration of consecutive years that meet the threshold
    - Subplot for each region
    - Overlapping events are not counted
    - Shows durations of events from 2 to 10 years in a row.
        - 1-year events are excluded to help y-axis scaling.
        - Assumes 9 years is the maximum duration (checked manually for each dataset).
    - Can use with both model and AGCD data.
    - Very similar to `plot_duration_histogram_downsampled`, but it this version
    is slightly faster than calling 'gsr_events.get_event_duration_counts' and
    then plotting a bar graph.
    """

    # Lowest number of consecutive years to show on x-axis
    min_duration = 2
    # Histogram bin edges (assumes 9-years is the max duration)
    bins = np.arange(min_duration - 0.5, 10)

    # Get labelled events
    _, ds_events = get_gsr_events(
        dv.pr,
        dv[quantile_var],
        threshold=event.threshold,
        min_duration=min_duration,
        operator=event.operator,
        fixed_duration=False,
        time_dim=time_dim,
    )

    if model != "AGCD":
        # Stack modelled events across ensemble & init_date
        ds_events = ds_events.rename({"event": "ev"})  # tmp rename
        ds_events = ds_events.stack({"event": ["init_date", "ensemble", "ev"]})

    # Check we're not excluding any long events
    max_duration = ds_events.duration.max().load()
    assert max_duration <= bins.max(), "Max duration exceeds histogram bins"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        ax.set_title(f"{letters[i]}) {model}: {regions[i]} region", loc="left")
        ax.hist(
            ds_events.duration.isel(x=i),
            bins=bins,
            color="b",
            edgecolor="k",
            align="mid",
        )

        ax.set_xlabel(f"Number of consecutive {event.name} {quantile_var} years")
        ax.set_ylabel("Frequency")
        ax.set_xmargin(1e-3)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_duration_histogram_{model}.png",
    )
    plt.show()


def plot_duration_histogram_downsampled(event, n_resamples=10000, density=True):
    """Histogram of the number of consecutive years that meet the threshold

    Shows a histogram of event durations for AGCD data with box and whisker
    overlays to show the downsampled distribution for each model when
    downsampled to match the number of samples in AGCD data.

    Parameters
    ----------
    model : str
        Name of the model to plot ("all" for show all models with AGCD bars).
    event : Events instance
        Event properties.
    n_resamples : int, default 10000
        Number of resamples for downsampling.
    density : bool, default True
        If True, plot probability (%) instead of frequency (slightly different
        sample sizes between models and observations).
    """

    # Model whisker colors
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(models) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for i, ax in enumerate(axes):
        ax.set_title(
            f"{letters[i]}) {regions[i]} region consecutive {event.alt_name} years",
            loc="left",
        )
        # Plot histogram bars for AGCD
        m = "AGCD"
        ds = gsr_data_regions(m).isel(x=i, drop=True)
        # Count number of events of each duration
        counts_agcd, bins, N_agcd = get_event_duration_counts(event, ds, m, density)

        ax.bar(
            bins[:-1],
            counts_agcd,
            color="b",
            edgecolor=None,
            width=1,
            label=m,
            alpha=0.3,
        )

        # Downsampled model boxplots (fits all models within each bar)
        for m, model in enumerate(models):

            ds = gsr_data_regions(model).isel(x=i)
            # Produces n_resamples of obs-sized subsample
            counts, bins, _ = get_event_duration_counts(
                event,
                ds,
                model,
                density,
                downsample=True,
                n_resamples=n_resamples,
                N_obs=N_agcd,
            )

            # Model boxplot
            ax.boxplot(
                counts,
                whis=[5, 95],
                sym=".",
                label=model,
                positions=(bins[:-1] - 0.5) + (m + 1) / 10,
                tick_labels=bins[:-1],
                widths=0.05,
                boxprops=dict(lw=0.7),
                whiskerprops=dict(color=colors[m]),
                medianprops=dict(color=colors[m]),
                flierprops=dict(
                    ms=4, markerfacecolor=colors[m], markeredgecolor=colors[m]
                ),
            )
        ax.set_xticks(bins[:-1])
        ax.set_xlim(bins[0] - 0.5, bins[-1] - 0.5)
        ax.set_xticklabels(bins[:-1])
        ax.set_xlabel(f"Number of consecutive {event.name} {quantile_var} years")

        if density:
            ax.set_ylabel("Probability (%)")
        else:
            ax.set_ylabel("Frequency")
        lgd = ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    outfile = f"{fig_dir}/{event.type}/{event.type}_duration_histogram_downsampled{'' if density else '_frequency'}.png"
    plt.savefig(
        outfile,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.show()


def plot_duration_heatmap(event):
    """Heatmap of the durations of consecutive years (draft).

    Parameters
    ----------
    event : Events instance
        Event properties.
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    for i, ax in enumerate(axes):
        ax.set_title(
            f"{letters[i]}) {regions[i]} region consecutive {event.alt_name} years",
            loc="left",
        )
        # model boxplots
        counts_list = []
        for m, model in enumerate(dataset_names):
            ds = gsr_data_regions(model).isel(x=i)
            counts, bins, _ = get_event_duration_counts(
                event,
                ds,
                model,
                density=True,
                downsample=False,
            )
            counts_list.append(counts)
        # combine model counts into (duration, model) xr.dataarray
        counts = xr.DataArray(
            np.stack(counts_list[::-1]),  # reverse to match dataset_names order
            dims=["model", "duration"],
            coords={"duration": bins[:-1], "model": dataset_names},
        )
        counts.attrs["units"] = "Probability (%)"
        bounds = np.power(10.0, np.arange(-3, 1))
        bounds = np.append(
            bounds, np.arange(2, 16 + 2, 2)
        )  # add upper bound to include max value
        ncolors = len(bounds) - 1
        cmap = plt.cm.get_cmap("plasma", ncolors)
        norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)
        # cmap = plt.cm.plasma
        # cmap.set_bad("lightgrey")
        # # use log scale for better visibility of small values
        # norm = mpl.colors.LogNorm(vmin=0.1, vmax=counts.max())
        counts.where(counts != 0).plot(ax=ax, cmap=cmap, norm=norm)

        ax.set_xticks(bins[:-1])
        ax.set_xlim(bins[0] - 0.5, bins[-1] - 0.5)
        ax.set_xticklabels(bins[:-1])
        ax.set_xlabel(f"Number of consecutive {event.name} {quantile_var} years")
        ax.set_ylabel("")

    plt.tight_layout()
    outfile = f"{fig_dir}/{event.type}/{event.type}_duration_heatmap.png"
    plt.savefig(
        outfile,
        bbox_inches="tight",
    )
    plt.show()


def plot_transition_duration_histogram(da_quantile, model, event, time_dim):
    """Histogram of time between a n-year dry (wet) event and wet (dry) year.

    Parameters
    ----------
    da_quantile : xarray.DataArray
        GSR timeseries converted to quantiles (regions on 'x' axis).
    model : str
        Name of the model to show in the title.
    event : Events instance
        Event properties.
    time_dim : str, default "time"
        Name of the core time dimension (in which to look for events).
    """

    transition = ["dry", "wet"]
    if event.operator == "greater":
        transition = transition[::-1]

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    fig.suptitle(
        f"{model} years between n {transition[0]} years and a {transition[1]} year"
    )

    counter = 0
    for i, region in enumerate(regions):
        for j, n in enumerate([1, 2, 3]):
            k, bins = transition_time(da_quantile, n, time_dim, transition[0])

            if model != "AGCD":
                k = k.sum(["init_date", "ensemble"])

            k = k.where(k > 0, drop=True)
            bins = bins[bins <= (k.years.max().item() + 1)]

            ax[i, j].bar(
                bins[1:],
                k.isel(x=i),
                width=1,
                align="center",
                color="b",
                edgecolor="k",
            )
            ax[i, j].set_title(
                f"{letters[counter]}) {region} region after {n}-year event", loc="left"
            )
            ax[i, j].set_xlabel("Duration [years]")
            ax[i, j].set_ylabel("Frequency")
            ax[i, j].set_xmargin(1e-3)
            ax[i, j].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax[i, j].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            counter += 1
    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_duration_histogram_{model}.png",
    )
    plt.show()


def subplot_transition_histogram(ax, k, total, bins, var, alpha=1):
    """Format a subplot with a histogram of next year GSR quantile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance to plot on.
    k : xarray.DataArray
        Transition counts.
    total : int
        Total number of events.
    bins : array-like
        Histogram bin edges.
    var : {'tercile', 'decile', 'decile_binned'}
        Name of the quantile variable to use for event detection.
    """

    if not isinstance(total, int):
        total = int(total.load().item())
    x = np.arange(len(bins) - 1)

    # Sample colors from the BrBG colormap
    colors = quantile_cmap(return_colors=True)

    # Plot histogram bars
    ax.bar(
        x,
        (k / total) * 100,
        width=1,
        align="edge",
        color=colors,
        edgecolor="k",
        alpha=alpha,
    )
    ax.axhline(100 / len(x), ls="--", color="k", label="Expected")
    ax.set_xticks(x + 0.5)
    ax.set_ylabel("Probability (%)")
    ax.yaxis.labelpad = 2

    if len(bins) == 4:
        ax.set_xticklabels(["Dry", "Average", "Wet"])
    else:
        ax.set_xticklabels(np.arange(1, len(bins)))
    ax.set_xlabel(f"Next year GSR {var.replace('binned_', '')}")

    # Add total event count to top left corner of each subplot
    ax.text(
        0.96,
        0.95,
        f"Events={total}",
        bbox=dict(fc="white", alpha=0.8),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=10,
    )

    return ax


def plot_transition_histogram(
    da_quantile, model, event, time_dim="time", quantile_var="tercile"
):
    """Plot histogram of next year GSR quantile (after 1,2,3-year spells).

    Parameters
    ----------
    da_quantile : xarray.DataArray
        GSR timeseries converted to quantiles (regions on 'x' axis).
    model : str
        Name of the model to show in the title.
    event : Events instance
        Event properties.
    time_dim : str, default "time"
        Name of the core time dimension (in which to look for events).
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.
    """

    # Calculate the transition counts & event totals
    ds = xr.Dataset()
    ds["k"], ds["total"], bins = transition_probability(
        da_quantile,
        event.threshold,
        event.operator,
        min_duration=np.arange(1, 4),
        var=quantile_var,
        time_dim=time_dim,
        binned=True,
    )
    # Sum over other dimensions
    dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
    ds = ds.sum(dims)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    title = f"{model} {event.name} {quantile_var.replace('binned_', '')} GSR events"
    if quantile_var == "decile":
        title += f" (Apr-Oct rain {event.decile})"
    plt.suptitle(title)

    for j, N in enumerate([1, 2, 3]):
        for i, region in enumerate(regions):
            ax[i, j].set_title(f"{region} region GSR after {N}yr event")
            dx = ds.isel(x=i, n=j)

            ax[i, j] = subplot_transition_histogram(
                ax[i, j], dx.k, dx.total, bins, quantile_var
            )

    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_histogram_{quantile_var}_{model}.png",
    )
    plt.show()


def plot_transition_histogram_downsampled(
    event, n_resamples=10000, quantile_var="tercile"
):
    """Plot next year GSR tercile histograms (AGCD with downsampled model box & whiskers).

    Parameters
    ----------
    event : Events instance
        Event properties.
    n_resamples : int, default 10000
        Number of resamples for downsampling.
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.
    """

    dss = []
    for model in models:
        dv = gsr_data_regions(model)
        ds = downsampled_transition_probability(
            dv[quantile_var], event, regions, target="AGCD", n_resamples=n_resamples
        )
        dss.append(ds.assign_coords(model=model))
    ds = xr.concat(dss, dim="model")
    ds["k_agcd"] = ds.k_agcd.isel(model=0, drop=True)
    ds["total"] = ds.total.isel(model=0, drop=True)

    bins = np.arange(1, 5)
    x = np.arange(len(bins) - 1)

    dims = [d for d in ds.k.dims if d not in ["n", "x", "sample", "q", "model"]]
    ds = ds.sum(dims)
    # Model whisker colors
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(models) + 1))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    counter = 0
    for i, r in enumerate(regions):
        for j, N in enumerate([1, 2, 3]):

            title = f"{letters[counter]}) {r} region: after {N} {event.alt_name} years"
            if j == 0:
                title = title[:-1]  # remove 's' for 1 year
            ax[i, j].set_title(title, loc="left")
            dx = ds.isel(x=i, n=j)

            # AGCD histogram bars (decrease bar transparency)
            ax[i, j] = subplot_transition_histogram(
                ax[i, j], dx.k_agcd, dx.total, bins, "", alpha=0.7
            )

            # Model boxplot
            for m in range(len(models)):
                ax[i, j].boxplot(
                    (dx.k.isel(model=m).T / dx.isel(model=m).total) * 100,
                    whis=[5, 95],
                    positions=x + (m + 1) / 10,
                    widths=0.03,
                    sym=".",
                    boxprops=dict(lw=0.4),
                    whiskerprops=dict(color=colors[m]),
                    flierprops=dict(ms=2, markeredgecolor=colors[m]),
                )
            ax[i, j].set_xticks(x + 0.5)
            if j != 0:
                ax[i, j].set_ylabel(None)  # only left column
            ax[i, j].set_xticklabels(["Dry", "Average", "Wet"])
            ax[i, j].set_xlim(0, 3)
            ax[i, j].set_ylim(-1, 105)
            counter += 1
    lines = [
        mpl.lines.Line2D([0], [0], label=m, color=c) for m, c in zip(models, colors)
    ]
    lgd = fig.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        fontsize=12,
        ncols=6,
        title=f"Models ({n_resamples} subsamples)",
    )
    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_histogram_downsampled.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.show()


def plot_transition_pie_chart(
    da_quantile, model, event, time_dim="time", quantile_var="tercile"
):
    """Plot pie chart of next year GSR quantile after n years.

    Rows are regions, columns are n-year events.

    Parameters
    ----------
    da_quantile : xarray.DataArray
        GSR timeseries converted to quantiles (regions on 'x' axis).
    model : str
        Name of the model to show in the title.
    event : Events instance
        Event properties.
    time_dim : str, default "time"
        Name of the core time dimension (in which to look for events).
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.
    """

    # Calculate the transition counts & event totals
    ds = xr.Dataset()
    ds["k"], ds["total"], _ = transition_probability(
        da_quantile,
        event.threshold,
        event.operator,
        np.arange(1, 4),
        var="tercile",
        time_dim=time_dim,
    )

    # Sum over other dimensions
    dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
    ds = ds.sum(dims)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    plt.suptitle(
        f"{model} transition probability after {event.name} {quantile_var} GSR events"
    )
    for i, region in enumerate(regions):
        for j, N in enumerate([1, 2, 3]):
            ax[i, j].set_title(f"{region} region after {N}yr event")
            wedges, _, _ = ax[i, j].pie(
                ds.isel(x=i, n=j).k,
                colors=quantile_cmap(return_colors=True),
                startangle=90,
                shadow=True,
                autopct="%1.1f%%",
                wedgeprops={"edgecolor": "k"},
            )
        # Add legends at the ends of the rows
        ax[i, 2].legend(
            wedges,
            ["Dry ", "Average", "Wet"],
            title=f"Next year {quantile_var}",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_pie_{model}.png",
    )
    plt.show()


def subplot_transition_probability_matrix(
    ax, da_quantile, event, time_dim, ds=None, show_hatching=True
):
    """Format a subplot TPM."""

    x = np.arange(3)
    cmap, norm, cbar_ticks = transition_probability_cmap(n_intervals=24)

    if ds is None:
        ds = xr.Dataset()
        ds["k"], ds["total"], _ = transition_probability(
            da_quantile,
            event.threshold,
            event.operator,
            x + 1,
            var="tercile",
            time_dim=time_dim,
        )
    if "p" not in ds:
        # Sum over other dimensions
        dims = [d for d in ds.k.dims if d not in ["n", "q", "x", "sample"]]
        ds = ds.sum(dims)
        ds["p"] = (ds.k / ds.total) * 100

    if "mask" not in ds and show_hatching:
        # Calculate the 95% confidence interval
        ci0, ci1 = binom_ci(ds.total, p=1 / 3)
        ds["mask"] = ds.p.where((ds.k < ci0) | (ds.k > ci1))

    # Plot probability heatmap
    cm = ax.pcolormesh(ds.p, cmap=cmap, norm=norm)

    # Add hatching where significant
    if show_hatching:
        ax.pcolor(
            ds.mask,
            cmap=mpl.colors.ListedColormap(["none"]),
            hatch=".",
            ec="k",
            zorder=1,
            lw=0,
        )
    ax.set_yticks(x + 0.5, x + 1)
    ax.set_xticks(x + 0.5, ["Dry", "Average", "Wet"])

    return ax, cm, cbar_ticks, ds


def plot_transition_probability_matrix(da_quantile, model, event, time_dim):
    """Plot the transition matrix for the next year GSR quantile."""

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True, sharey=True
    )
    for i, region in enumerate(regions):
        ax = axes[i]
        ax.set_title(f"{model} {region} region transitions")
        ax, cm, cbar_ticks, ds = subplot_transition_probability_matrix(
            ax, da_quantile.isel(x=i), event, time_dim
        )

        ax.set_xlabel("Next year Apr-Oct rainfall")

    axes[0].set_ylabel(f"Consecutive {event.name} tercile years")
    cbar = fig.colorbar(
        cm,
        ax=[ax],
        label="Probability (%)",
        shrink=0.95,
        format="%3.1f",
        drawedges=False,
        ticks=cbar_ticks,
    )
    cbar.set_ticks(
        cbar_ticks[2::2], labels=[f"{tick:.1f}" for tick in cbar_ticks[2::2]]
    )
    cbar.ax.tick_params(size=0, length=0)
    # plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_matrix_{model}.png",
    )
    plt.show()


def plot_transition_probability_matrix_mmm(event):
    """Plot the transition matrix for the next year GSR quantile.

    (a-b) AGCD in each region
    (c-d) Multi-model median in each region
    """

    fig, ax = plt.subplots(2, 2, figsize=[12, 10], constrained_layout=True, sharey=True)
    for i, region in enumerate(regions):
        ax[0, i].set_title(
            f"{letters[i]}) {region} region {dataset_names[0]} transitions",
            loc="left",
        )
        ax[1, i].set_title(
            f"{letters[i+2]}) {region} region multi-model median transitions",
            loc="left",
        )

        dv = gsr_data_regions(dataset_names[0])
        ax[0, i], cm, cbar_ticks, ds = subplot_transition_probability_matrix(
            ax[0, i], dv.tercile.isel(x=i), event, "time"
        )

        # Multi-model median
        ds_list = []
        for model in models:
            dv = gsr_data_regions(model)
            ds = xr.Dataset()
            ds["k"], ds["total"], _ = transition_probability(
                dv.tercile.isel(x=i),
                event.threshold,
                event.operator,
                np.arange(1, 4),
                var="tercile",
                time_dim="lead_time",
            )
            # Sum over other dimensions
            dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
            ds = ds.sum(dims)
            ds["p"] = (ds.k / ds.total) * 100
            ds_list.append(ds.assign_coords(model=model))

        ds_mmm = xr.concat(ds_list, dim="model")
        ax[1, i], cm, cbar_ticks, _ = subplot_transition_probability_matrix(
            ax[1, i],
            ds_mmm,
            event,
            "lead_time",
            ds=ds_mmm.median(dim="model"),
            show_hatching=False,
        )
        ax[1, i].set_xlabel("Next year Apr-Oct rainfall")
        ax[i, 0].set_ylabel(f"Consecutive {event.name} tercile years")
        cbar = fig.colorbar(
            cm,
            ax=[ax[i, 1]],  # end of each row (switched)
            label="Probability (%)",
            shrink=0.95,
            format="%3.1f",
            drawedges=False,
            ticks=cbar_ticks,
        )
        cbar.set_ticks(
            cbar_ticks[2::2], labels=[f"{tick:.1f}" for tick in cbar_ticks[2::2]]
        )
        cbar.ax.tick_params(size=0, length=0)
    # plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_matrix_mmm.png",
    )
    plt.show()


def plot_transition_probability_matrix_combined(event):
    """Plot the transition matrix for the next year GSR quantile.

    Subplots grouped by # number of consecutive years, with datasets along
    y-axis and next year tercile along the x-axis.
    (a-c) WA region:, (d-f) SA region
    """

    # Concat datasets (obs, multi-model median, indiv models)
    ds_list = []
    for model in dataset_names:
        dv = gsr_data_regions(model)
        ds = xr.Dataset()
        ds["k"], ds["total"], _ = transition_probability(
            dv.tercile,
            event.threshold,
            event.operator,
            np.arange(1, 4),
            var="tercile",
            time_dim="lead_time" if model in models else "time",
            binned=True,
        )
        # Sum over other dimensions (e.g., ensemble, init_date)
        dims = [d for d in ds.k.dims if d not in ["n", "q", "x", "sample"]]
        ds = ds.sum(dims)
        ds["p"] = (ds.k / ds.total) * 100
        # Calculate the 95% confidence interval
        ci0, ci1 = binom_ci(ds.total, p=1 / 3)
        ds["mask"] = ds.p.where((ds.k < ci0) | (ds.k > ci1))

        ds_list.append(ds.assign_coords(model=model))

    # Multimodel median
    ds_mmm = xr.concat(ds_list[1:], dim="model")
    ds_mmm = ds_mmm.median("model").assign_coords(model="MMM")
    ds_mmm["mask"] *= np.nan  # no hatching for MMM
    # Insert the MMM after obs
    ds_list.insert(1, ds_mmm)
    # Merge datasets
    ds = xr.concat(ds_list, dim="model")
    # Reverse order of datasets (so obs is at top of y-axis)
    ds = ds.isel(model=slice(None, None, -1))
    # Add attrs
    ds["q"].attrs["description"] = "Next year tercile"
    ds["n"] = ds["n"] + 1
    ds["n"].attrs["description"] = "Number of consecutive years"
    cmap, norm, cbar_ticks = transition_probability_cmap(n_intervals=24)

    # ds.p.plot(col='n', col_wrap=3, figsize=(12, 7), cmap=cmap, norm=norm, shading='flat')

    fig, ax = plt.subplots(
        len(regions),
        3,
        figsize=(13, 10),
        layout="constrained",
        # sharex=True,
        sharey=True,
    )

    counter = 0
    for x, region in enumerate(regions):
        for i, n in enumerate(ds.n.values):

            pcm = ax[x, i].pcolormesh(ds.p.isel(n=i, x=x), cmap=cmap, norm=norm)
            ax[x, i].pcolor(
                ds.mask.isel(n=i, x=x),
                cmap=mpl.colors.ListedColormap(["none"]),
                hatch=".",
                ec="k",
                zorder=1,
                lw=0,
            )
            # Center tercile labels on x-axis
            ax[x, i].set_xticks(np.arange(ds.q.size) + 0.5, ["Dry", "Average", "Wet"])
            # Center model names between grids on y-axis
            ax[x, i].set_yticks(
                np.arange(ds.model.size) + 0.5, [m for m in ds.model.values]
            )
            title = f"{letters[counter]}) {region}: Transition after {n} {event.alt_name} years"
            if n == 1:
                title = title[:-1]  # fix plural ('years' => 'year')
            ax[x, i].set_title(title, loc="left")
            counter += 1

    # add horizontal colourbar below subplots
    # cax = ax[0].inset_axes([0.3, -0.1, 1.5, 0.05])
    cbar = fig.colorbar(
        pcm,
        ax=ax[x, :],
        # cax=cax,
        label="Probability (%)",
        orientation="horizontal",
        shrink=0.9,
        pad=0.05,
        aspect=30,
    )
    cbar.set_ticks(
        cbar_ticks[2::2], labels=[f"{tick:.1f}" for tick in cbar_ticks[2::2]]
    )

    cbar.ax.tick_params(size=0, length=0)
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_matrix.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_transition_probability_matrix_all_models(event, quantile_var="tercile"):
    """Plot the transition matrix for the next year GSR quantile."""

    # Plot each region a separate plot
    for i, region in enumerate(regions):

        fig, axes = plt.subplots(3, 3, figsize=[14, 12], sharey=True, sharex=True)
        ax = axes.flatten()
        fig.suptitle(f"{region} region transition probabilities", y=0.93)

        for m, model in enumerate(models):
            dv = gsr_data_regions(model).isel(x=i)

            ax[m].set_title(f"{letters[m]}) {model}", loc="left")
            ax[m], cm, cbar_ticks, ds = subplot_transition_probability_matrix(
                ax[m], dv.tercile, event, "lead_time"
            )
            if m >= 6:
                # Add xlabel to the bottom of each column
                ax[m].set_xlabel("Next year Apr-Oct rainfall")
            if m % 3 == 0:
                # Add label at the start of each row
                ax[m].set_ylabel(f"Consecutive {event.name} {quantile_var} years")

        # add cbar at the bottom of the figure that stretches along cols
        cax = fig.add_axes([0.25, 0.01, 0.5, 0.03])
        cbar = fig.colorbar(
            cm,
            ax=axes[-1, :],
            cax=cax,
            label="Probability (%)",
            orientation="horizontal",
            shrink=0.9,
            pad=0.0,
            aspect=20,
        )
        cbar.set_ticks(
            cbar_ticks[2::2], labels=[f"{tick:.1f}" for tick in cbar_ticks[2::2]]
        )
        cbar.ax.tick_params(size=0, length=0)
        plt.savefig(
            f"{fig_dir}/{event.type}/{event.type}_transition_matrix_all_models.png",
            bbox_inches="tight",
        )
        plt.show()


def plot_transition_probability_matrix_downsampled(
    da_quantile, model, event, quantile_var="tercile"
):
    """Plot 10 random examples of model TPMs after downsampling.

    Creates a separate figure for each region.

    Parameters
    ----------
    da_quantile : xarray.DataArray
        GSR timeseries converted to quantiles (regions on 'x' axis).
    model : str
        Name of the model to show in the title.
    event : Events instance
        Event properties.
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.
    """

    # Get samples of the downsampled transition matrices
    time_dim = "lead_time"
    nr, nc = 5, 5
    n_resamples = nr * nc

    ds = downsampled_transition_probability(
        da_quantile, event, regions, target="AGCD", n_resamples=n_resamples
    )

    # Plot nr samples of transition histograms at a region
    for i, r in enumerate(regions):

        fig, axes = plt.subplots(nr, nc, figsize=(16, 14), sharey=True)
        axes = axes.flatten()
        plt.suptitle(
            f"{r} region: {model} {event.name} {quantile_var} GSR event probability of transition (%) (same event sample sizes as AGCD)"
        )
        for s in range(nr * nc):
            ax = axes[s]
            ax, cm, cbar_ticks, _ = subplot_transition_probability_matrix(
                ax, da_quantile.isel(x=i), event, time_dim, ds.isel(x=i, sample=s)
            )
            ax.set_title(f"Random sample #{s + 1}", fontsize=11, loc="left")
            if s >= nc * (nr - 1):
                # Add xlabel to the bottom of each column
                ax.set_xlabel("Next year Apr-Oct rainfall", fontsize=11)
            if s % nc == 0:
                # At label at the start of each row
                ax.set_ylabel(
                    f"Consecutive {event.name} {quantile_var} years", fontsize=11
                )
            cbar = plt.colorbar(
                cm,
                ax=ax,
                shrink=0.95,
                format="%3.1f",
            )
            cbar.set_ticks(
                cbar_ticks[2::2], labels=[f"{tick:.1f}" for tick in cbar_ticks[2::2]]
            )
            cbar.ax.tick_params(labelsize=8, size=0, length=0)

        plt.tight_layout()
        plt.savefig(
            f"{fig_dir}/{event.type}/{event.type}_transition_matrix_downsampled_{r}_{model}.png",
        )
        plt.show()


def plot_transition_sample_size(
    da_quantile, model, event, n_resamples=10000, quantile_var="tercile"
):
    """Transition probabilities histograms with different sample sizes.

    Parameters
    ----------
    da_quantile : xarray.DataArray
        GSR timeseries converted to quantiles (regions on 'x' axis).
    model : str
        Name of the model to show in the title.
    event : Events instance
        Event properties.
    n_resamples : int, default 10000
        Number of resamples for downsampling.
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.

    Notes
    -----
    - Creates a 3x2 subplot figure with rows as N-year events (1,2,3) and
    columns as regions (WA, SA).
    - Each subplot contains box & whisker plots of the transition probabilities
    for different event sample sizes.
    - 3 box & whisker plots per sample size (dry, average, wet).
    - 5 sample sizes: 5, 10, 100, 1000,
    """

    colors = quantile_cmap(return_colors=True)
    # Target event sample sizes (x-axis shows 3 transition probabilities for each size)
    targets = 10 ** np.arange(4)
    targets[0] = 5
    xticklabels = [r"$10^{}$".format(i) for i, _ in enumerate(targets)]
    xticklabels[0] = str(5)
    event_durations = [1, 2, 3]

    fig, ax = plt.subplots(
        len(event_durations),
        len(regions),
        figsize=(11, 9),
        constrained_layout=True,
        sharey=True,
    )
    fig.suptitle(f"{model} {event.name} {quantile_var} Apr-Oct rainfall events")

    # Iterate over regions (columns)
    for xi, x in enumerate(regions):

        # Iterate over sample size (within each subplot)
        for k, s in enumerate(targets):

            ds = downsampled_transition_probability(
                da_quantile, event, regions, target=s, n_resamples=n_resamples
            )

            #  Iterate over N year events (rows)
            for ni, n in enumerate(event_durations):
                counter = xi + (len(regions) * ni)  # subplot counter for titles
                title = f"{letters[counter]}) {x} region: transition after {n} {event.alt_name} years"
                if n == 1:
                    title = title[:-1]  # remove 's' for 1 year
                ax[ni, xi].set_title(title, loc="left")

                # Plot line at expected probability (33.3%)
                ax[ni, xi].axhline(100 / 3, c="k", ls=(0, (5, 10)), lw=0.7)
                bp = ax[ni, xi].boxplot(
                    ds.p.isel(x=xi, n=ni).T,
                    whis=[5, 95],
                    positions=k + np.array([0.75, 1, 1.25]),
                    widths=0.2,
                    sym=".",
                    flierprops=dict(ms=2),
                    patch_artist=True,
                )

                # Fill boxes with colors
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

                ax[ni, xi].margins(x=0.1)
                ax[ni, xi].set_xticks(np.arange(len(targets)) + 1)
                ax[ni, xi].set_xticklabels(xticklabels)
                ax[ni, xi].set_xlabel("Total number of events")
                ax[ni, 0].set_ylabel("Probability (%)")  # Only left column
                ax[ni, xi].legend(
                    bp["boxes"], ["Dry", "Average", "Wet"], loc="upper right"
                )
    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_transition_sample_size_{model}.png",
    )
    plt.show()


def plot_timeseries_AGCD(dv, var="pr", anom=False, trend=False):
    """Plot timseries of AGCD GSR (rainfall or quantiles) in WA & SA regions.

    Parameters
    ----------
    dv : xarray.Dataset
        GSR data for AGCD regions.
    var : {'pr', 'tercile', 'decile'}, default 'tercile'
        Name of the variable to plot.
    anom : bool, default=False
        Plot anomalies (relative to the mean) instead of absolute values.
    trend : bool, default=False
        Plot linear trendline and test for significance (Mann-Kendall test).
    """

    # Plot GSR
    filename_str = ""
    da = dv[var]
    if anom:
        da = da - da.mean("time")
        filename_str += "_anom"
    if trend:
        filename_str += "_trend"
    ylabel = "Rainfall [mm]"

    _, axes = plt.subplots(2, 1, figsize=(10, 5))

    # Iterate for each region
    for i, state in enumerate(["Western Australia", "South Australia"]):
        axes[i].set_title(
            f"{letters[i]}) AGCD Apr-Oct rainfall in the {state} region", loc="left"
        )

    # Iterate for each region
    for i, ax in enumerate(axes):
        # Plot da timeseries as bars
        ax.bar(dv.pr.time.dt.year, da.isel(x=i), color="blue")

        # Add linear trend line and print result of significance test
        if trend:
            # Test significance of trend
            x = dv.pr.time.dt.year.astype(int)
            y = da.isel(x=i)
            mask = ~np.isnan(y)
            # Mann-Kendall test (non-parametric test for monotonic trend)
            res = pymannkendall.original_test(y.where(mask))
            print(res)

            # Plot linear trend line (solid line if p-value < 0.05)
            trend_line = np.arange(len(y)) * res.slope + res.intercept
            if res.p >= 0.05:
                # Not significant
                ls = "dashed"
                label = "Trend"
            else:
                ls = "solid"
                label = "Trend (p < 0.05)"
            ax.plot(x, trend_line, color="red", lw=2, ls=ls, label=label)
            ax.legend()

        ax.set_ylabel(ylabel)
        ax.set_xmargin(0)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/timeseries_AGCD{filename_str}.png",
    )
    plt.show()


def plot_timeseries_AGCD_events(dv, event, quantile_var="tercile"):
    """Plot the AGCD (rainfall & quantile) timeseries with GSR events shaded.

    Parameters
    ----------
    dv : xarray.Dataset
        GSR data for AGCD regions.
    event : Events instance
        Event properties.
    quantile_var : {'tercile', 'decile'}, default 'tercile'
        Name of the quantile variable to use for event detection.
    """

    # Find dry/wet spells (non-overlapping)
    events, _ = get_gsr_events(
        dv.pr,
        dv[quantile_var],
        threshold=event.threshold,
        min_duration=event.n,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        time_dim="time",
    )

    # Plot GSR rainfall or GSR quantile with rainfall
    fnames = [f"_{quantile_var}", ""]
    da_list = [dv[quantile_var], dv.pr]
    ylabels = [f"GSR {quantile_var}", "Rainfall [mm]"]

    if quantile_var == "tercile":
        # Don't plot pr with tercile events shaded
        da_list = da_list[:1]

    for k, da in enumerate(da_list):
        _, axes = plt.subplots(2, 1, figsize=(10, 5))
        for i, state in enumerate(["Western Australia", "South Australia"]):
            axes[i].set_title(
                f"{letters[i]}) AGCD Apr-Oct rainfall in the {state} region", loc="left"
            )

        for j, ax in enumerate(axes):
            # Plot timeseries as bars
            ax.bar(dv.pr.time.dt.year, da.isel(x=j), color="blue")

            # Plot the quantile threshold (if plotting quantiles)
            if da.max() <= 10:
                ax.axhline(event.threshold, c="k", lw=0.5)

            # Shade the GSR periods
            for i in sorted(np.unique(events.isel(x=j)))[1:]:
                t = da.time.dt.year[events.isel(x=j).load() == i]
                ax.axvspan(t[0] - 0.33, t[-1] + 0.3, color="red", alpha=0.3)

            ax.set_ylabel(ylabels[k])
            ax.set_xmargin(0)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        plt.tight_layout()
        plt.savefig(
            f"{fig_dir}/{event.type}/timeseries_{event.type}{event.type[4:]}_{event.n}yr{fnames[k]}_AGCD.png",
        )
        plt.show()


def plot_timeseries_DCPP(dv, event, model):
    """Plot the DCPP tercile timeseries with GSR events shaded."""

    # Get the GSR events
    events, _ = get_gsr_events(
        dv.pr,
        dv.tercile,
        threshold=event.threshold,
        min_duration=event.n,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        time_dim="lead_time",
    )

    # Plot init_dates along columns and ensemble members along rows
    nr = dv.ensemble.size
    init_dates = np.arange(dv.init_date.size)
    nc = len(init_dates)
    year0 = dv.tercile.time.isel(lead_time=0).dt.year.min().item()

    # Create a plot for each region
    for x, region in enumerate(regions):  # WA or SA region
        fig, ax = plt.subplots(
            nr, nc, figsize=(1.8 * nc, 1.2 * nr), sharey=True, sharex="col"
        )
        fig.suptitle(
            f"{model} {region} region {event.n}yr events of GSR {event.tercile}",
        )

        # Iterate through ensemble members
        for j in range(nr):
            # Iterate through init_dates
            for i, init_date in enumerate(init_dates[:nc]):
                loc = {"ensemble": j, "init_date": init_date, "x": x}
                da = dv.tercile.isel(loc)
                da = da.where(~np.isnan(da), drop=True)
                years = da.time.dt.year.astype(dtype=int)

                # Plot bars
                ax[j, i].bar(years, da, color="b", label=j + 1)
                ax[j, i].axhline(event.threshold, c="k", lw=0.5)

                # Highlight event periods
                for k in sorted(np.unique(events.isel(loc)))[1:]:
                    inds = np.nonzero(events.isel(loc).values == k)[0]
                    t = years.isel(lead_time=inds)
                    ax[j, i].axvspan(t[0] - 0.4, t[-1] + 0.4, color="red", alpha=0.3)

                ax[j, i].set_xmargin(0)
                ax[j, i].set_xticks([years[k] for k in [0, 5 if len(years) > 5 else 3]])
                ax[j, i].set_yticks([0, 5, 10])
                ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
                ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            ax[j, 0].set_ylabel("Tercile")

            # Add ensemble number in subplot
            ax[j, 0].text(
                year0,
                10,
                f"e{j}",
                ha="left",
                va="top",
                size="small",
                bbox=dict(fc="white", alpha=0.7),
            )

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.2, wspace=0.17)
        plt.savefig(
            f"{fig_dir}/timeseries_{event.type}{event.type[4:]}_{region}_{event.n}yr_{model}.png",
        )
        plt.show()


def plot_timeseries_heatmap(da):
    """Plot timeseries heatmap of DCPP ensemble terciles (draft)."""

    # ALt timeseries plot for larger ensembles (terciles as heatmap)
    cmap, norm = quantile_cmap(return_colors=False)
    da.isel(ensemble=slice(3)).plot(
        col="ensemble", cmap=cmap, norm=norm, figsize=(12, 10)
    )
    da.isel(x=0).plot(
        col="ensemble",
        cmap=cmap,
        norm=norm,
        figsize=(30, 20),
        col_wrap=12,
        cbar_kwargs=dict(
            orientation="horizontal", fraction=0.01, pad=0.05, ticks=[1, 2, 3]
        ),
    )
    plt.savefig(f"{home}/test.png")


def plot_stability_pdfs(model, da, start_years=np.arange(1960, 2020, 10)):
    """Stability assessment PDFs for a model in WA and SA regions.

    Plots distributions by lead time and by start year.
    2x2 subplot: top row lead time, bottom row start year, columns are regions.

    Parameters
    ----------
    model : str
        Name of the model to show in the title.
    da : xarray.DataArray
        GSR rainfall DataArray for the model (regions on 'x' axis).
    start_years : array-like, default np.arange(1960, 2020, 10)
        Start years to use when plotting distributions by time.
    """

    metric = "Apr-Oct rainfall"
    units = "Apr-Oct Rainfall [mm / day]"

    lead_dim = "lead_time"
    dims = ["ensemble", "init_date", lead_dim]
    da = da.dropna(lead_dim, how="all").stack({"sample": dims})

    if da.time.dtype != "O":
        # Convert time coords to cftime (if not already)
        times_new = da["time"].dt.strftime(f"%Y-%m-%d")
        times_new = np.vectorize(str_to_cftime)(times_new)
        da["time"] = (da["time"].dims, times_new)
        da["time"].attrs = da.time.attrs

    fig, ax = plt.subplots(2, 2, figsize=[20, 17])
    for i, r in enumerate(regions):
        plot_dist_by_lead(
            ax[0, i], da.isel(x=i), metric, units=units, lead_dim=lead_dim
        )
        ax[0, i].set_title(
            f"({letters[i]}) {model} distribution by lead ({r} region)",
        )
        plot_dist_by_time(ax[1, i], da.isel(x=i), metric, start_years, units=units)
        ax[1, i].set_title(
            f"({letters[i + 2]}) {model} distribution by year ({r} region)",
        )

    outfile = f"{fig_dir}/stability-test-pdfs_growing-season-pr_{model}.png"
    plt.savefig(outfile, bbox_inches="tight", facecolor="white")


def plot_stability_multimodel(
    stability_dim="time", start_years=np.arange(1960, 2020, 10)
):
    """Plot each models stability assessment plot in each region.

    Parameters
    ----------
    stability_dim : {'time', 'lead'}, default 'time'
        Dimension along which to plot distributions.
    start_years : array-like, default np.arange(1960, 2020, 10)
        Start years to use when plotting distributions by time.

    Notes
    -----
    - Seperate plot for each region
    - 3x3 subplot: each model in a subplot
    - Distributions by lead time or by start year
    """

    metric = "Apr-Oct rainfall"
    units = "Apr-Oct Rainfall [mm / day]"

    lead_dim = "lead_time"
    dims = ["ensemble", "init_date", lead_dim]

    for i, r in enumerate(regions):
        fig, ax = plt.subplots(3, 3, figsize=[25, 17])
        ax = ax.flatten()
        for m, model in enumerate(models):
            dv = gsr_data_regions(model).isel(x=i).pr
            da = dv.dropna(lead_dim, how="all").stack({"sample": dims})
            if da.time.dtype != "O":
                # Convert time coords to cftime (if not already)
                times_new = da["time"].dt.strftime(f"%Y-%m-%d")
                times_new = np.vectorize(str_to_cftime)(times_new)
                da["time"] = (da["time"].dims, times_new)
                da["time"].attrs = da.time.attrs

            if stability_dim == "lead":
                plot_dist_by_lead(ax[m], da, metric, units=units, lead_dim=lead_dim)
                ax[m].set_title(
                    f"({letters[m]}) {model} distribution by lead ({r} region)",
                )
            elif stability_dim == "time":
                plot_dist_by_time(ax[m], da, metric, start_years, units=units)
                ax[m].set_title(
                    f"({letters[m]}) {model} distribution by year ({r} region)",
                )

        outfile = f"{fig_dir}/stability-test-{stability_dim}_growing-season-pr-{r}.png"
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches="tight", facecolor="white")
        plt.show()


if __name__ == "__main__":

    n_resamples = 10000
    quantile_var = "tercile"

    # Dataset-specific plots
    for model in dataset_names:
        for operator in ["less", "greater"]:
            time_dim = "time" if model == "AGCD" else "lead_time"
            # Properties of events with different upper limits
            event = Events(n=3, operator=operator)
            event_max = Events(2, operator, fixed_duration=False)
            events = [Events(i, operator) for i in [2, 3]]
            dv = gsr_data_regions(model)

            # Region specific plots
            if model in models:
                plot_stability_pdfs(model, dv.pr)
                # plot_timeseries_DCPP(dv, event, model)
                plot_transition_probability_matrix_downsampled(dv.tercile, model, event)
                plot_transition_sample_size(dv.tercile, model, event, n_resamples)
            else:
                plot_transition_probability_matrix(dv.tercile, model, event, time_dim)
                plot_timeseries_AGCD(dv, anom=True, trend=True)
                for ev in [event_max, *events]:
                    plot_timeseries_AGCD_events(dv, ev)

            plot_transition_probability_matrix(dv.tercile, model, event, time_dim)
            plot_duration_histogram(
                dv, event_max, model, time_dim=time_dim, quantile_var="tercile"
            )
            plot_transition_histogram(dv.tercile, model, event, time_dim)
            plot_transition_pie_chart(dv.tercile, model, event, time_dim)

            plot_transition_duration_histogram(dv.tercile, model, event, time_dim)

    # Multi-model plots
    for event in [Events(operator="less"), Events(operator="greater")]:
        plot_transition_probability_matrix_all_models(event)
        plot_transition_probability_matrix_mmm(event)
        plot_transition_histogram_downsampled(event, n_resamples)
        plot_duration_histogram_downsampled(event, n_resamples)
        plot_duration_histogram_downsampled(event, n_resamples, density=True)
    plot_transition_probability_matrix_combined(event)
    plot_stability_multimodel(stability_dim="time")

    # Print longest event durations for each model
    for model in models:
        time_dim = "time" if model == "AGCD" else "lead_time"
        dv = gsr_data_regions(model)
        duration = []
        for event in [Events(operator="less"), Events(operator="greater")]:
            min_duration = 1
            _, ds_events = get_gsr_events(
                dv.pr,
                dv.tercile,
                threshold=event.threshold,
                min_duration=min_duration,
                operator=event.operator,
                fixed_duration=False,
                time_dim=time_dim,
            )
            ds = ds_events.stack({"sample": ["init_date", "ensemble", "event"]})
            duration.append(ds.duration.max("sample").astype(dtype=int).values)
        print(
            f"{model:<14s} (WA, SA): dry={duration[0][0]},{duration[0][1]} wet={duration[1][0]},{duration[1][1]}"
        )
