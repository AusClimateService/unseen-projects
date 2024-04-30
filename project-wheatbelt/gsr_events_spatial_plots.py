"""Low/high growing season (Apr-Oct) rainfall event maps."""

import cartopy.crs as ccrs
import cmocean
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import xarray as xr

from unseen.general_utils import plot_aus_map
from gsr_events import (
    home,
    models,
    Events,
    get_AGCD_data_au,
    get_events_au,
    get_DCPP_data_au,
    transition_probability,
    binom_ci,
)


def plot_shapefile(ax):
    """Plot shapefiles of the South Australia and Western Australia regions."""
    regions = gp.read_file(home / "shapefiles/crops_SA_WA.shp")
    ax.add_geometries(
        regions.geometry,
        ccrs.PlateCarree(),
        lw=0.4,
        facecolor="none",
        edgecolor="r",
        zorder=1,
    )
    return ax


def plot_stippling(ax, p_mask, model):
    """Add stippling where values are significant."""
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


def plot_timeseries(ds, data, decile, lat, lon):
    """Plot timeseries with event shading at a grid point."""
    # lat, lon = -12.4, 136.55
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.bar(data.time.dt.year, decile.sel(lat=lat, lon=lon), color="b")
    # Shade GSR periods
    dv = ds.sel(lat=lat, lon=lon).load().dropna("event")
    for i, f in zip(
        dv.index_start.astype(dtype=int).values, dv.index_end.astype(dtype=int).values
    ):
        t = data.time.dt.year[i : f + 1]
        ax.axvspan(t[0] - 0.33, t[-1] + 0.3, color="red", alpha=0.3)


def plot_frequency(ds, model, event, n):
    """Plot map of event frequency (per 100 years)."""
    da = ds.id.count("event")
    da = da * 100 / n  # per 100 years
    vmax = da.max()
    levels = MaxNLocator(nbins=vmax * 4 if vmax < 5 else vmax).tick_values(0, vmax)
    cmap = cmocean.cm.thermal
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 7), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    ax = plot_shapefile(ax)
    ax = plot_aus_map(
        fig,
        ax,
        da,
        title=f"{model} Apr-Oct rainfall {event.name}",
        outfile=home
        / f"figures/{event.event_type}_map_frequency_{model}_{event.min_duration}yr.png",
        cbar_kwargs=dict(
            fraction=0.05, extend="max", label="Frequency (per 100 years)"
        ),
        cmap=cmap,
        norm=norm,
    )


def plot_duration(ds, model, event):
    """Plot maps of the median and maximum duration of events."""
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 7),
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    cmap = cmocean.cm.thermal
    # Colourbar limits and number of bins
    vlim = [[2, 4], [2, 12]]
    nbins = [4, 10]

    for i, da in enumerate(
        [ds.duration.median("event"), ds.duration.max("event")],
    ):
        levels = MaxNLocator(nbins=nbins[i]).tick_values(vlim[i][0], vlim[i][1])
        ax[i] = plot_shapefile(ax[i])
        ax[i] = plot_aus_map(
            fig,
            ax[i],
            da,
            title=f"{['Median', 'Maximum'][i]} number of consecutive years {event.decile}",
            cbar_kwargs=dict(
                label="Duration [years]",
                orientation="horizontal",
                fraction=0.034,
                extend="max",
                pad=0.05,
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )
    st = fig.suptitle(
        f"{model} Apr-Oct rainfall {event.decile}",
        ha="center",
        va="bottom",
        y=0.8,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig(
        home
        / f"figures/{event.event_type[:4]}_map_duration_{model}_{event.min_duration}yr.png",
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
    for i, var, vmax in zip(range(3), ["min", "mean", "max"], [1200, 1200, 1600]):
        levels = MaxNLocator(nbins=10).tick_values(0, vmax)
        cmap = cmocean.cm.rain
        ax[i] = plot_shapefile(ax[i])
        ax[i] = plot_aus_map(
            fig,
            ax[i],
            ds["gsr_" + var].mean("event").load(),
            title=f"{model} {var} GSR ({event.name})",
            cbar_kwargs=dict(
                label="Apr-Oct rainfall [mm]",
                orientation="horizontal",
                fraction=0.034,
                extend="max",
                pad=0.04,
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )
    plt.tight_layout(pad=0.3)
    plt.savefig(
        home / f"figures/{event.event_type}_map_pr_{model}_{event.min_duration}yr.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def persistance_probability_map(decile, event, model, time_dim="time"):
    """Transition probability map for 1 year decile events."""
    k, n, _ = transition_probability(
        decile,
        event.threshold,
        event.operator,
        min_duration=1,
        time_dim=time_dim,
        binned=True,
    )
    k = k.isel(q=0, drop=True)
    dims = [d for d in k.dims if d not in ["lat", "lon"]]
    if len(dims) > 0:
        k = k.sum(dims)
        n = n.sum(dims)
    p = k / n

    # Mask where p-value is not significant.
    # pvalue = apply_binomtest(k, n, p=0.3, alternative="two-sided")
    # p_mask = p.where((pvalue <= alpha) | (pvalue >= (1 - alpha)))
    ci0, ci1 = binom_ci(n, p=0.3)
    p_mask = p.where((k < ci0) | (k > ci1))

    levels = MaxNLocator(nbins=9).tick_values(0, 90)
    cmap = plt.cm.PuOr
    cmap = cmocean.tools.crop_by_percent(cmap, 30, which="min", N=None)

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 7), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    # Plot SA and WA regions
    ax = plot_shapefile(ax)

    # Add stippling where significant
    ax = plot_stippling(ax, p_mask, model)

    ax = plot_aus_map(
        fig,
        ax,
        p * 100,
        title=f"{model} Apr-Oct rainfall {event.decile} transition probability",
        cbar_kwargs=dict(fraction=0.05, extend="max", label="Probability [%]"),
        cmap=cmap,
        norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
    )

    plt.tight_layout()
    plt.savefig(
        home
        / f"figures/{event.event_type[:4]}_map_persistance_probability_{model}.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()


def transition_probability_maps(decile, event, model, time_dim="time"):
    """Transition probability map for n year decile events."""
    k, n, _ = transition_probability(
        decile,
        threshold=event.threshold,
        operator=event.operator,
        min_duration=event.min_duration,
        time_dim=time_dim,
        binned=True,
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
        f"{model} Apr-Oct rainfall {event.decile} for {event.min_duration} consecutive years",
        y=0.5,
    )
    # Plot maps of transition probabilities to a dry, medium, and wet year
    for i, ax in enumerate(axes.flat):
        # Plot SA and WA regions
        ax = plot_shapefile(ax)

        # Calculate confidence intervals & plot stippling where significant
        # N.B. adjust expected probabilty as the "medium" bin includes deciles 4-7 (40% of data)
        # whereas there are 3 deciles the wet/dry bins (each 30% of data)
        ci0, ci1 = binom_ci(n, p=0.4 if i == 1 else 0.3)
        p_mask = p.isel(q=i).where((k.isel(q=i) < ci0) | (k.isel(q=i) > ci1))
        ax = plot_stippling(ax, p_mask, model)

        # Adjust colormap centre to highlight expect probability
        cmap = plt.cm.PuOr
        cmap = cmocean.tools.crop_by_percent(
            cmap, 20 if i == 1 else 30, which="min", N=None
        )
        levels = MaxNLocator(nbins=11).tick_values(0, 100)

        ax = plot_aus_map(
            fig,
            ax,
            p.isel(q=i) * 100,
            title=f"Probability the following year will be {['dry', 'medium', 'wet'][i]}",
            cbar_kwargs=dict(
                orientation="horizontal",
                fraction=0.034,
                pad=0.04,
                label="Probability [%]",
                extend="max",
            ),
            cmap=cmap,
            norm=BoundaryNorm(levels, ncolors=cmap.N, clip=True),
        )

    plt.tight_layout()
    plt.savefig(
        home
        / f"figures/{event.event_type[:4]}_map_transition_prob_{model}_{event.min_duration}yr.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()


if __name__ == "__main__":
    for model in models:
        if model == "AGCD":
            time_dim = "time"
            data, decile = get_AGCD_data_au()
            n = data.time.count("time")  # Total samples (for frq plot)
        else:
            time_dim = "lead_time"
            data, decile = get_DCPP_data_au(model)
            n = data["ensemble"].size * data["init_date"].size * data["lead_time"].size

        for operator in ["less", "greater"]:

            # Properties of events with no upper limit (for maximum duration plot)
            event_max = Events(min_duration=2, operator=operator, fixed_duration=False)
            ds_max = get_events_au(data, decile, event_max, model, time_dim)

            # Stack n=2,3 year event property datasets along dimension "n"
            evs = [Events(i, operator=operator, minimize=True) for i in [2, 3]]
            ds = [get_events_au(data, decile, evs[i], model, time_dim) for i in [0, 1]]
            ds = xr.concat([ds[i].expand_dims({"n": i + 1}) for i in [0, 1]], dim="n")

            if model != "AGCD":
                # Stack DCPP model event properties
                ds = ds.rename({"event": "ev"})
                ds = ds.stack({"event": ["init_date", "ensemble", "ev"]})
                ds_max = ds_max.rename({"event": "ev"})
                ds_max = ds_max.stack({"event": ["init_date", "ensemble", "ev"]})

            # Plot transition probability map for 1-year decile events
            persistance_probability_map(decile, Events(1, operator), model, time_dim)

            # Transition probability maps for n-year events (dry, medium & wet transition maps)
            for event in [Events(i + 1, operator) for i in range(3)]:
                transition_probability_maps(decile, event, model, time_dim)

            # Median and maximum duration of consecutive years of high/low GSR
            plot_duration(ds_max, model, event_max)

            # Plot spatial maps for each n-year event duration
            for i, event in enumerate(evs):
                # Plot frequency n-year events (# of events per 100 years)
                plot_frequency(ds.isel(n=i, drop=True), model, event, n=n)
                # Maps of minimum, median, and maximum GSR during events
                plot_event_stats(ds.isel(n=i, drop=True), model, event)
