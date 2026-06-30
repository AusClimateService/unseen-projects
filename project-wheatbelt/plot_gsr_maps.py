"""Plots maps of low/high growing season (Apr-Oct) rainfall events."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
import cmocean
from cmocean.tools import crop_by_percent
from collections import defaultdict
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MaxNLocator
import numpy as np
import xarray as xr

from process_gsr_data import (
    home,
    fig_dir,
    dataset_names,
    models,
    gsr_data_aus_AGCD,
    gsr_data_aus_DCPP,
    gsr_data_regions,
)
from gsr_events import Events, get_gsr_events_gridded, binom_ci, transition_probability

plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300


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
    cbar_kwargs : dict, optional
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

    if isinstance(data, xr.DataArray):
        cs = ax.pcolormesh(data.lon, data.lat, data, zorder=0, **kwargs)
        fig.colorbar(cs, **cbar_kwargs)

    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3, zorder=1)
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
        plt.savefig(outfile, bbox_inches="tight")
        plt.show()
    return ax


def plot_shapefile(
    ax, gdf=None, edgecolor="k", facecolor="none", lw=0.6, ls="-", **kwargs
):
    """Plot shapefile outlines of the South Australia and Western Australia regions.

    Parameters
    ----------
    ax : matplotlib plot axis
    gdf : geopandas GeoDataFrame, optional
        GeoDataFrame of the shapefile to plot. If None, default shapefile is used
    edgecolor : str, optional
        Color of the shapefile region edges. Default is 'k' (black).
    facecolor : str, default 'none'
        Color of the shapefile region faces. Use 'none' for no fill.
    """

    if gdf is None:
        gdf = gp.read_file(home / "shapefiles/crops_SA_WA.shp")

    ax.add_geometries(
        gdf.geometry,
        ccrs.PlateCarree(),
        lw=lw,
        ls=ls,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=1,
        **kwargs,
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
    ax = plot_shapefile(ax, edgecolor="white")

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
        f"{fig_dir}/{event.type}/{event.type}_map_count_{event.n}yr_{model}.png",
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
    ax = plot_shapefile(ax, edgecolor="white")
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
        f"{fig_dir}/{event.type}/{event.type}_map_frequency_{event.n}yr_{model}.png",
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

        ax[i] = plot_shapefile(ax[i], edgecolor="white")
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
        f"{fig_dir}/{event.type}/{event.type}_map_duration_{model}.png",
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
        ax[i] = plot_shapefile(ax[i], edgecolor="grey", ls="-")
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
        f"{fig_dir}/{event.type}/{event.type}_map_pr_{event.n}yr_{model}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_deciles(dv, model, q=10):
    """Plot maps of Apr-Oct rainfall deciles."""

    dims = ["ensemble", "init_date", "lead_time"] if model != "AGCD" else "time"

    bins = dv.pr.quantile(q=np.arange(q + 1) / q, dim=dims)

    fig, ax = plt.subplots(
        1, 3, figsize=(14, 8), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    fig.suptitle(f"{model}", y=0.42)
    for i, decile in enumerate([3, 5, 8]):

        levels = MaxNLocator(nbins=10).tick_values(0, 1000)
        cmap = cmocean.cm.rain
        ax[i] = plot_shapefile(ax[i], edgecolor="grey", ls="-")
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
        f"{fig_dir}/map_deciles_{model}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_persistance_probability(tercile, event, model, time_dim="time"):
    """Transition probability map for 1 year quantile events."""

    k, n, _ = transition_probability(
        tercile,
        event.threshold,
        event.operator,
        min_duration=1,
        var="tercile",
        time_dim=time_dim,
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
    ax = plot_shapefile(ax, edgecolor="grey")

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
            label="Probability of transition (%)",
        ),
        cmap=cmap,
        norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
    )
    ax.set_extent([112, 154, -44, -25], crs=ccrs.PlateCarree())

    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/{event.type}/{event.type}_map_persistance_probability_{model}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_transition_probability(tercile, event, model, time_dim="time"):
    """Transition probability map for n year quantile events."""

    k, n, _ = transition_probability(
        tercile,
        threshold=event.threshold,
        operator=event.operator,
        min_duration=event.n,
        var="tercile",
        time_dim=time_dim,
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
        ax = plot_shapefile(ax, edgecolor="grey")

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
                label="Probability of transition (%)",
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
        f"{fig_dir}/{event.type}/{event.type}_map_transition_probability_{event.n}yr_{model}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_study_region_maps():
    """Map of australia with insets for WA and SA regions."""

    def add_model_grid_lines(ax, extent, dv, model, lw=0.5):
        """Add model grid lines (cell boundaries) to an axis."""

        # dv = gsr_data_aus_DCPP(model)
        lats, lons = dv.lat, dv.lon
        lon_edges = lons + np.gradient(lons, 2)
        lat_edges = lats + np.gradient(lats, 2)

        ax.hlines(
            lat_edges,
            xmin=extent[0],
            xmax=extent[1],
            color="grey",
            linewidth=lw,
            transform=ccrs.PlateCarree(),
        )
        ax.vlines(
            lon_edges,
            ymin=extent[2],
            ymax=extent[3],
            color="grey",
            linewidth=lw,
            transform=ccrs.PlateCarree(),
            label=f"{model} grid",
        )
        return ax

    # Colors for each region
    COLOR_WA = "#E63946"
    COLOR_SA = "b"  # "#457B9D"

    gdf_aus = gp.read_file(home / "shapefiles/australia.shp")
    gdf_wa = gp.read_file(home / "shapefiles/crops_WA.shp")
    gdf_sa = gp.read_file(home / "shapefiles/crops_SA.shp")
    bounds_wa = gdf_wa.total_bounds
    bounds_sa = gdf_sa.total_bounds

    extent = [109, 158, -45, -10.2]  # [west, east, south, north]
    pad = 0.5
    extent_wa = np.around(
        [
            bounds_wa[0] - pad + 0.08,
            bounds_wa[2] + pad - 0.08,
            bounds_wa[1] - pad,
            bounds_wa[3] + pad - 0.3,
        ],
        1,
    )
    extent_sa = np.around(
        [
            bounds_sa[0] - pad - 0.05,
            bounds_sa[2] + pad + 0.05,
            bounds_sa[1] - pad - 0.3,
            bounds_sa[3] + pad + 0.4,
        ],
        1,
    )

    ds = [gsr_data_aus_DCPP(models[4]), gsr_data_aus_DCPP(models[0])]

    fig = plt.figure(figsize=(10, 8))
    ax = [None] * 3
    # Main Australia map (left, bottom, width, height)
    ax[0] = fig.add_subplot(
        2, 1, 1, projection=ccrs.PlateCarree(), position=[0.4, 0.4, 0.2, 0.4]
    )
    ax[1] = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax[2] = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())

    # ax[0].set_title("(a) Study Regions Overview", loc="left")
    ax[1].set_title("(a) Western Australia (WA) Region")
    ax[2].set_title("(b) South Australia (SA) Region")

    # Australia map with WA and SA regions highlighted (upper middle)
    ax[0].add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
    # ax[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax[0].add_feature(cfeature.LAND, alpha=0.5)
    ax[0].set_extent(extent, crs=ccrs.PlateCarree())
    ax[0] = plot_shapefile(
        ax[0], gdf_aus, "k", facecolor="none"
    )  # , alpha=0.1, lw=1.5)
    ax[0] = plot_shapefile(
        ax[0], gdf_wa, COLOR_WA, facecolor=COLOR_WA, alpha=0.4, lw=1.5
    )
    ax[0] = plot_shapefile(
        ax[0], gdf_sa, COLOR_SA, facecolor=COLOR_SA, alpha=0.4, lw=1.5
    )
    rect_wa = mpatches.Rectangle(
        (extent_wa[0], extent_wa[2]),
        extent_wa[1] - extent_wa[0],
        extent_wa[3] - extent_wa[2],
        fill=False,
        edgecolor=COLOR_WA,
        linewidth=1.5,
        linestyle="-",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    rect_sa = mpatches.Rectangle(
        (extent_sa[0], extent_sa[2]),
        extent_sa[1] - extent_sa[0],
        extent_sa[3] - extent_sa[2],
        fill=False,
        edgecolor=COLOR_SA,
        linewidth=1.5,
        linestyle="-",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax[0].add_patch(rect_wa)
    ax[0].add_patch(rect_sa)

    # (a) WA region with HadGEM3 gridlines
    ax[1] = plot_aus_map(fig, ax[1], None)
    ax[1] = plot_shapefile(
        ax[1], gdf_wa, COLOR_WA, facecolor=COLOR_WA, alpha=0.4, lw=1.5
    )
    for spine in ax[1].spines.values():
        spine.set_edgecolor(COLOR_WA)
        spine.set_linewidth(1.75)
    ax[1].set_xticks(np.arange(110, 145, 4))
    ax[1].set_yticks(np.arange(-34, -26, 2))
    ax[1].set_extent(extent_wa, crs=ccrs.PlateCarree())

    # (b) SA region with CAFE gridlines
    ax[2] = plot_aus_map(fig, ax[2], None)
    ax[2] = plot_shapefile(
        ax[2], gdf_sa, COLOR_SA, facecolor=COLOR_SA, alpha=0.4, lw=1.5
    )
    for spine in ax[2].spines.values():
        spine.set_edgecolor(COLOR_SA)
        spine.set_linewidth(1.75)
    ax[2].set_xticks(np.arange(132, 145, 4))
    ax[2].set_yticks(np.arange(-38, -26, 2))
    ax[2].set_extent(extent_sa, crs=ccrs.PlateCarree())

    # (b, c): Add model grid lines and legend
    ax[1] = add_model_grid_lines(ax[1], extent_wa, ds[0], models[4], lw=0.8)
    ax[2] = add_model_grid_lines(ax[2], extent_sa, ds[1], models[0], lw=0.8)
    ax[1].legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax[2].legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax[0].axes.set_axis_off()

    plt.savefig(f"{fig_dir}/regions_map_alt_0.png", bbox_inches="tight")
    plt.show()


def combine_figures(files, outfile, axes, axis=False):
    """Combine plotted figures of single models into a image."""

    files = sorted(files)

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

    fig_path = home / "figures"

    for subfolder in ["LGSR", "HGSR"]:
        path = fig_path / subfolder

        # Get and sort files by length (longest first) to prioritize specific patterns
        files_all = sorted(path.glob("*.png"), key=lambda f: len(f.name), reverse=True)
        subgroups = defaultdict(list)
        assigned_files = set()

        for f in files_all:
            if f.name in assigned_files:
                continue
            # Generate the pattern key (stripping the model/suffix)
            prefix = f.stem.rsplit("_", 1)[0]
            filestem = f"{prefix}"

            # Find all matching files for this specific prefix that aren't assigned yet
            matches = []
            for m in dataset_names:
                f = path / f"{prefix}_{m}.png"
                if f.exists() and f.name not in assigned_files:
                    matches.append(f)

            if matches:
                subgroups[filestem] = matches
                assigned_files.update([m.name for m in matches])

        # Plot subgroups
        for filestem, files in subgroups.items():
            outfile = f"{path}/{filestem}_combined.png"

            # Combine as 3x3 grid
            if len(subgroups[filestem]) == 9:
                _, axes = plt.subplots(3, 3, figsize=[12, 10], layout="compressed")
                combine_figures(files, outfile, axes, axis=True)

            # Combine as 2x5 grid
            elif len(subgroups[filestem]) == 10:
                _, axes = plt.subplots(5, 2, figsize=[8, 6], layout="compressed")
                combine_figures(files, outfile, axes, axis=True)


if __name__ == "__main__":
    """Generate all GSR event maps for each model."""

    plot_study_region_maps()

    # Iterate through datasets and low or high GSR spells
    for model in dataset_names:
        for operator in ["less", "greater"]:

            time_dim = "time" if model == "AGCD" else "lead_time"

            # Properties of events with different upper limits
            event = Events(n=3, operator=operator)
            event_max = Events(2, operator, fixed_duration=False)
            events = [Events(i, operator) for i in [1, 2, 3]]

            dv = gsr_data_regions(model)

            # Map plots
            if model == "AGCD":
                dv = gsr_data_aus_AGCD()
                n_times = dv.pr.time.count("time").load().item()
            else:
                dv = gsr_data_aus_DCPP(model)
                n_times = (~np.isnan(dv.tercile)).sum(
                    ["init_date", "ensemble", "lead_time"]
                )
                n_times = n_times.max().load().item()

            ds_max = get_gsr_events_gridded(
                dv.pr,
                dv.tercile,
                time_dim,
                threshold=event_max.threshold,
                min_duration=event_max.min_duration,
                fixed_duration=event_max.fixed_duration,
                operator=event_max.operator,
            )

            # Stack n=2,3 year event property datasets along dimension "n"
            evs = [Events(i, operator=operator, minimize=False) for i in [2, 3]]
            ds = [
                get_gsr_events_gridded(
                    dv.pr,
                    dv.tercile,
                    time_dim,
                    kwargs=dict(
                        threshold=evs[i].threshold,
                        min_duration=evs[i].min_duration,
                        fixed_duration=evs[i].fixed_duration,
                        operator=evs[i].operator,
                    ),
                )
                for i in [0, 1]
            ]
            ds = xr.concat(
                [ds[i].assign_coords({"n": i + 2}) for i in [0, 1]],
                dim="n",
                join="outer",
            )

            if model != "AGCD":  # Stack DCPP model event properties
                ds = ds.rename({"event": "tmp"})
                ds = ds.stack({"event": ["init_date", "ensemble", "tmp"]})

            # Plot maps
            plot_deciles(dv, model)

            # Plot spatial maps for each n-year event duration
            for i, event in enumerate(evs):
                plot_event_count(
                    ds.isel(n=i, drop=True), dv.tercile, model, event, n_times
                )
                # Plot frequency n-year events (# of events per 100 years)
                plot_frequency(ds.isel(n=i, drop=True), model, event, n_times)
                # Maps of minimum, median, and maximum GSR during events
                plot_event_stats(ds.isel(n=i, drop=True), model, event)

            # Median and maximum duration of consecutive years of high/low GSR
            plot_duration(ds_max, model, event_max)

            # Plot transition probability map for 1-year quantiles events
            plot_persistance_probability(
                dv.tercile, Events(1, operator), model, time_dim
            )

            # Transition probability maps for n-year events (dry, medium & wet transition maps)
            for event in [Events(i + 1, operator) for i in range(3)]:
                plot_transition_probability(dv.tercile, event, model, time_dim)
    combine_all_figures()
