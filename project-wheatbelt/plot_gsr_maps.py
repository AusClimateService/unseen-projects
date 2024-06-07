"""Plots maps of low/high growing season (Apr-Oct) rainfall events."""

import cartopy.crs as ccrs
import cmocean
from cmocean.tools import crop_by_percent
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, FixedLocator
import numpy as np

from process_gsr_data import home
from gsr_events import transition_probability
from utils import binom_ci, plot_aus_map

plt.rcParams["font.size"] = 13


def plot_shapefile(ax, color, ls="-"):
    """Plot shapefile outlines of the South Australia and Western Australia regions."""

    regions = gp.read_file(home / "shapefiles/crops_SA_WA.shp")
    ax.add_geometries(
        regions.geometry,
        ccrs.PlateCarree(),
        lw=0.6,
        ls=ls,
        facecolor="none",
        edgecolor=color,
        zorder=1,
    )
    return ax


def plot_map_stippling(ax, p_mask, model):
    """Plot stippling (dots) where values are significant."""
    ax.pcolor(
        p_mask.lon,
        p_mask.lat,
        p_mask,
        cmap=mpl.colors.ListedColormap(["none"]),
        hatch="..",
        ec="k",
        transform=ccrs.PlateCarree(),
        zorder=1,
        lw=5e-4 if model == "AGCD" else 0,
    )
    return ax


def plot_event_count(ds, data, model, event, n_times):
    """Plot Australian map of event frequency (per 100 years)."""
    n_events = ds.id.count("event")
    if model == "AGCD":
        mask = ~np.isnan(data.isel(time=0, drop=True))
    else:
        mask = ~np.isnan(data.isel(ensemble=0, lead_time=0, init_date=0, drop=True))
    n_events = n_events.where(mask)  # Mask zero events for plotting

    fig = plt.figure(figsize=(10, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_shapefile(ax, color="white")

    # Add the total time count to the bottom left corner
    ax.text(
        0.04,
        0.08,
        f"Total years={n_times}",
        bbox=dict(fc="white", alpha=0.5),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )
    cmap = plt.cm.inferno
    levels = MaxNLocator(nbins=10).tick_values(n_events.min(), n_events.max())

    ax = plot_aus_map(
        fig,
        ax,
        n_events,
        title=f"{model} number of {event.n}-year {event.decile} GSR events",
        cbar_kwargs=dict(fraction=0.05, label="Total events"),
        cmap=cmap,
        norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
    )
    ax.set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type}_map_count_{event.n}yr_{model}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_frequency(ds, model, event, n_times):
    """Plot Australian map of event frequency (per 100 years)."""
    # Discrete colour map (for 2 and 3 year events)
    cmap = cmocean.cm.thermal
    vlim = [1, 10] if event.n == 2 else [0, 4.5]
    levels = MaxNLocator(nbins=9).tick_values(vlim[0], vlim[1])

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 4), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    ax = plot_shapefile(ax, color="white")
    # Add the total time count to the bottom left corner
    ax.text(
        0.04,
        0.08,
        f"Total years={n_times}",
        bbox=dict(fc="white", alpha=0.5),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax = plot_aus_map(
        fig,
        ax,
        (ds.id.count("event") * 100) / n_times,
        title=f"{model} frequency of {event.decile} GSR for {event.n} years in a row",
        cbar_kwargs=dict(
            fraction=0.04, extend="max", label="Frequency (per 100 years)"
        ),
        cmap=cmap,
        norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
    )
    ax.set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())
    plt.tight_layout(pad=0.5)
    plt.savefig(
        home / f"figures/{event.type}_map_frequency_{event.n}yr_{model}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_duration(ds, model, event):
    """Plot maps of the median and maximum duration of events."""
    if model != "AGCD":
        ds = ds.rename({"event": "ev"})
        ds = ds.stack({"event": ["init_date", "ensemble", "ev"]})

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    cmap = cmocean.cm.thermal
    levels = [
        MaxNLocator(nbins=4).tick_values(2, 4),
        MaxNLocator(nbins=10).tick_values(2, 10),
    ]

    for i, da in enumerate(
        [ds.duration.median("event"), ds.duration.max("event")],
    ):

        ax[i] = plot_shapefile(ax[i], color="white")
        ax[i] = plot_aus_map(
            fig,
            ax[i],
            da,
            title=f"{['Median', 'Maximum'][i]} consecutive years {event.decile}",
            cbar_kwargs=dict(
                label="Duration [years]",
                orientation="horizontal",
                fraction=0.06,
                extend="max",
                pad=0.12,
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels[i], ncolors=cmap.N, clip=True),
        )
        ax[i].set_xticks(np.arange(120, 155, 10))
        ax[i].set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    st = fig.suptitle(
        f"{model} Apr-Oct rainfall {event.decile}",
        ha="center",
        va="bottom",
        y=0.78,
    )

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.75)
    plt.savefig(
        home / f"figures/{event.type[:4]}_map_duration_{model}.png",
        dpi=200,
        bbox_inches="tight",
        bbox_extra_artists=[st],
    )
    plt.show()


def plot_event_stats(ds, model, event):
    """Plot maps of the min, mean, max Apr-Oct rainfall during events."""
    fig, ax = plt.subplots(
        1, 3, figsize=(14, 8), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    fig.suptitle(
        f"{model} {event.n}-year {event.decile} Apr-Oct rainfall events", y=0.42
    )
    for i, var, vmax in zip(range(3), ["min", "mean", "max"], [1200, 1200, 1600]):
        levels = MaxNLocator(nbins=10).tick_values(0, vmax)
        cmap = cmocean.cm.rain
        ax[i] = plot_shapefile(ax[i], color="grey", ls="-")
        ax[i] = plot_aus_map(
            fig,
            ax[i],
            ds["gsr_" + var].mean("event").load(),
            title=f"Event {var}",
            cbar_kwargs=dict(
                label="Apr-Oct rainfall [mm]",
                orientation="horizontal",
                fraction=0.034,
                extend="max",
                pad=0.05,
                aspect=30,
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )
        ax[i].set_xticks(np.arange(120, 155, 10))
        ax[i].set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    plt.tight_layout(pad=0.5)
    plt.savefig(
        home / f"figures/{event.type}_map_pr_{event.n}yr_{model}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_deciles(dv, model):
    """Plot maps of Apr-Oct rainfall deciles."""
    dims = ["ensemble", "init_date", "lead_time"] if model != "AGCD" else "time"
    q = np.arange(11)
    bins = dv.pr.quantile(q=q / (len(q) - 1), dim=dims)

    fig, ax = plt.subplots(
        1, 3, figsize=(14, 8), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    fig.suptitle(f"{model}", y=0.42)
    for i, decile in enumerate([3, 5, 8]):

        levels = MaxNLocator(nbins=10).tick_values(0, 1000)
        cmap = cmocean.cm.rain
        ax[i] = plot_shapefile(ax[i], color="grey", ls="-")
        ax[i] = plot_aus_map(
            fig,
            ax[i],
            bins.isel(quantile=decile).load(),
            title=f"Decile {decile} Apr-Oct rainfall",
            cbar_kwargs=dict(
                label="Apr-Oct rainfall [mm]",
                orientation="horizontal",
                fraction=0.034,
                extend="max",
                pad=0.05,
                aspect=30,
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )
        ax[i].set_xticks(np.arange(120, 155, 10))
        ax[i].set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    plt.tight_layout(pad=0.5)
    plt.savefig(
        home / f"figures/map_deciles_{model}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_persistance_probability(tercile, event, model, time="time"):
    """Transition probability map for 1 year decile events."""
    k, n, _ = transition_probability(
        tercile,
        event.threshold,
        event.operator,
        min_duration=1,
        var="tercile",
        time=time,
    )
    # Select dry-dry/wet-wet transitions
    q = 0 if event.operator == "less" else 2
    k = k.isel(q=q, drop=True)

    # Sum over all dimensions (for DCPP models)
    dims = [d for d in k.dims if d not in ["lat", "lon"]]
    if len(dims) > 0:
        k = k.sum(dims)
        n = n.sum(dims)

    # Probability: successful transitions / total transitions
    p = k / n

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 4), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    # Plot SA and WA regions
    ax = plot_shapefile(ax, color="grey")

    # Add stippling where significant
    ci0, ci1 = binom_ci(n, p=0.3)
    p_mask = p.where((k < ci0) | (k > ci1))
    ax = plot_map_stippling(ax, p_mask, model)

    cmap = crop_by_percent(plt.cm.PuOr, 60 - 100 / 3, which="min", N=None)
    levels = FixedLocator(np.arange(0, 1, 1 / 12) * 100).tick_values(0, 90)

    ax = plot_aus_map(
        fig,
        ax,
        p * 100,
        title=f"{model} {event.name} tercile Apr-Oct rain transition probability",
        cbar_kwargs=dict(
            fraction=0.05,
            extend="max",
            format="%3.1f",
            label="Probability of transition [%]",
        ),
        cmap=cmap,
        norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
    )
    ax.set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_map_persistance_probability_{model}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()


def plot_transition_probability(tercile, event, model, time="time"):
    """Transition probability map for n year decile events."""
    k, n, _ = transition_probability(
        tercile,
        threshold=event.threshold,
        operator=event.operator,
        min_duration=event.n,
        var="tercile",
        time=time,
    )
    dims = [d for d in k.dims if d not in ["q", "lat", "lon"]]
    if len(dims) > 0:
        k = k.sum(dims)
        n = n.sum(dims)
    p = k / n

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 10), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    fig.suptitle(
        f"{model} {event.n}-year {event.name} tercile Apr-Oct rainfall", y=0.36
    )
    # Plot maps of transition probabilities to a dry, medium, and wet year
    for i, ax in enumerate(axes.flat):
        # Plot SA and WA regions
        ax = plot_shapefile(ax, color="grey")

        # Calculate confidence intervals & plot stippling where significant
        ci0, ci1 = binom_ci(n, p=1 / 3)
        p_mask = p.isel(q=i).where((k.isel(q=i) < ci0) | (k.isel(q=i) > ci1))
        ax = plot_map_stippling(ax, p_mask, model)

        # Adjust colormap centre to highlight expected probability
        cmap = crop_by_percent(plt.cm.PuOr, 55 - 100 / 3, which="min", N=None)
        levels = FixedLocator(np.arange(0, 1, 1 / 12) * 100).tick_values(0, 90)

        ax = plot_aus_map(
            fig,
            ax,
            p.isel(q=i) * 100,
            title=f"Probability of {['dry', 'medium', 'wet'][i]} next year",
            cbar_kwargs=dict(
                orientation="horizontal",
                fraction=0.034,
                pad=0.04,
                label="Probability of transition [%]",
                extend="max",
                format="%3.1f",
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )
        ax.set_xticks(np.arange(120, 155, 10))
        ax.set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.savefig(
        home
        / f"figures/{event.type[:4]}_map_transition_probability_{event.n}yr_{model}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()
