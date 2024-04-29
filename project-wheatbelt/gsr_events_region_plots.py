"""Low/high growing season (Apr-Oct) rainfall event in the WA and SA regions."""

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr

from gsr_events import (
    home,
    models,
    Events,
    gsr_events,
    get_AGCD_data_regions,
    get_DCPP_data_regions,
    transition_probability,
    binom_ci,
)

regions = ["WA", "SA"]


def plot_AGCD_timeseries(data, decile, event):
    """Plot the AGCD (rainfall & decile) timeseries with GSR events shaded."""
    events, _ = gsr_events(
        data,
        decile,
        threshold=event.threshold,
        min_duration=event.min_duration,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        minimize=event.minimize,
        time_dim="time",
    )
    # Plot GSR rainfall or GSR decile
    for da, ylabel, fname in zip(
        [data, decile], ["Rainfall [mm]", "GSR decile"], ["", "_decile"]
    ):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        for i, state in enumerate(["Western Australia", "South Australia"]):
            axes[i].set_title(
                f"AGCD Apr-Oct rainfall in the {state} region", loc="left"
            )

        for j, ax in enumerate(axes):
            # Plot timeseries as bars
            ax.bar(data.time.dt.year, da.isel(x=j), color="blue", label="Rainfall Data")

            # Plot the decile threshold (if plotting deciles)
            if da.max() == 10:
                ax.axhline(event.threshold, c="k", lw=0.5)

            # Shade the GSR periods
            for i in sorted(np.unique(events.isel(x=j)))[1:]:
                t = da.time.dt.year[events.isel(x=j) == i]
                ax.axvspan(t[0] - 0.33, t[-1] + 0.3, color="red", alpha=0.3)

            ax.set_ylabel(ylabel)
            ax.set_xmargin(0)
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        plt.tight_layout()
        plt.savefig(
            home
            / f"figures/{event.event_type[:4]}_timeseries_AGCD{event.event_type[4:]}_{event.min_duration}yr{fname}.png",
            dpi=200,
        )
        plt.show()


def plot_DCPP_timeseries(data, deciles, event, model, n=0):
    """Plot the DCPP decile timeseries with GSR events shaded."""
    events, _ = gsr_events(
        data,
        deciles,
        threshold=event.threshold,
        min_duration=event.min_duration,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        minimize=event.minimize,
        time_dim="lead_time",
    )
    n = 0  # init_date subset to plot (max = len(init_date_start))
    init_date_start = np.arange(0, data.init_date.size + 6, 6)[n]

    for x, region in enumerate(regions):  # WA or SA region
        fig, ax = plt.subplots(10, 6, figsize=(14, 10), sharey=True, sharex="col")
        fig.suptitle(
            f"{model} {region} region {event.min_duration}yr events of GSR {event.decile}"
        )
        for j, ensemble in enumerate(range(10)):
            for i, init_date in enumerate(
                range(init_date_start, min(data.init_date.size, init_date_start + 6))
            ):
                loc = {"ensemble": j, "init_date": init_date, "x": x}
                decile = deciles.isel(loc)
                decile = decile.where(~np.isnan(decile), drop=True)
                years = decile.time.dt.year.astype(dtype=int)

                # Plot bars
                ax[j, i].bar(years, decile, color="b", label=j + 1)
                ax[j, i].axhline(event.threshold, c="k", lw=0.5)

                # Highlight event periods
                for k in sorted(np.unique(events.isel(loc)))[1:]:
                    t = years[events.isel(loc) == k]
                    ax[j, i].axvspan(t[0] - 0.4, t[-1] + 0.4, color="red", alpha=0.3)
                ax[j, i].set_xmargin(0)
                ax[j, i].set_xticks([years[k] for k in [0, -1]])
                ax[j, i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
                ax[j, i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

                if i == 0:
                    # Add ensemble number in subplot
                    ax[j, 0].text(
                        years[0], 10, f"e{j}", ha="left", va="top", size="xx-small"
                    )
                    ax[j, 0].set_ylabel("GSR decile")

        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.tight_layout()
        plt.savefig(
            home
            / f"figures/{event.event_type[:4]}_timeseries_{model}{event.event_type[4:]}_{region}_{event.min_duration}yr_decile_t{n}.png",
            dpi=200,
        )
        plt.show()


def next_year_decile_histograms(decile, model, event, time_dim, binned=False):
    """Plot the histogram of next year GSR deciles after n years."""
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    plt.suptitle(f"{model} GSR events {event.decile}")

    for j, min_duration in enumerate([1, 2, 3]):
        # Calculate the transition counts
        k, n, bins = transition_probability(
            decile,
            event.threshold,
            event.operator,
            min_duration,
            time_dim,
            binned=binned,
        )
        # Sum over other dimensions
        dims = [d for d in k.dims if d not in ["q", "x"]]
        if len(dims) > 0:
            k = k.sum(dims)

        for i, region in enumerate(regions):
            bin_width = np.diff(bins)
            ax[i, j].bar(
                bins[:-1],
                k.isel(x=i) / bin_width,
                width=np.diff(bins),
                align="edge",
                color="b",
                edgecolor="k",
            )
            ax[i, j].set_title(
                f"{region} region next year GSR (after {min_duration}yr event)",
                loc="left",
            )
            if binned:
                ax[i, j].set_xticks([2.5, 6, 9.5])
                ax[i, j].set_xticklabels(["Dry", "Medium", "Wet"])
                ax[i, j].set_ylabel("Frequency / bin")
                ax[i, j].set_xlabel("Decile range")
            else:
                ax[i, j].set_xticks(np.arange(1, 11) + 0.5)
                ax[i, j].set_xticklabels(np.arange(1, 11))
                ax[i, j].set_ylabel("Frequency")
                ax[i, j].set_xlabel("Decile")
    plt.tight_layout()

    name = f"figures/{event.event_type}_next_year_decile_hist_{model}.png"
    if binned:
        name = name.replace(".png", "_binned.png")
    plt.savefig(home / name, dpi=200)
    plt.show()


def transition_probability_matrix(decile, model, event):
    """Plot the transition matrix for the next year GSR decile."""
    K = []
    N = []
    for j, min_duration in enumerate([1, 2, 3]):
        k, n, bins = transition_probability(
            decile,
            event.threshold,
            event.operator,
            min_duration,
            time_dim,
            binned=True,
        )
        # Sum over other dimensions
        dims = [d for d in k.dims if d not in ["q", "x"]]
        if len(dims) > 0:
            k = k.sum(dims)
            n = n.sum(dims)
        N.append(n.assign_coords({"n": j}))
        K.append(k.assign_coords({"n": j}))

    k = xr.concat(K, dim="n")
    n = xr.concat(N, dim="n")

    p = (k / n) * 100

    p_mask = xr.DataArray(np.full_like(p, np.nan), coords=p.coords)
    for j in range(3):
        for x in range(2):
            for i, q in enumerate([0.3, 0.4, 0.3]):
                loc = dict(n=j, x=x, q=i)
                ci0, ci1 = binom_ci(n.isel(n=j, x=x, drop=True), p=q)
                p_mask[loc] = p.isel(loc).where(
                    (k.isel(loc) < ci0) | (k.isel(loc) > ci1)
                )

    x = np.arange(3)
    levels = MaxNLocator(nbins=9).tick_values(0, 90)
    cmap = plt.cm.PuOr
    cmap = cmocean.tools.crop_by_percent(cmap, 30, which="min", N=None)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True, sharey=True
    )
    for i, region in enumerate(regions):
        ax = axes[i]
        cm = ax.pcolormesh(p.isel(x=i), cmap=cmap, norm=norm)
        ax.set_xticks(x + 0.5)
        ax.set_yticks((x + 0.5))
        ax.set_yticklabels((x + 1))
        ax.set_xticklabels(["Dry", "Medium", "Wet"])
        axes[0].set_ylabel(f"Consecutive years of GSR {event.decile}")
        ax.set_xlabel("Decile range")
        ax.set_title(f"{model} {region} region next year GSR")

        ax.pcolor(
            p_mask.isel(x=i),
            cmap=mpl.colors.ListedColormap(["none"]),
            hatch=".",
            ec="k",
            zorder=1,
            lw=0,
        )
    fig.colorbar(cm, ax=[ax], label="Probability [%]", shrink=0.95, extend="max")
    plt.savefig(
        home / f"figures/{event.event_type[:4]}_transition_matrix_{model}.png", dpi=200
    )
    plt.show()


def max_duration_histogram(data, decile, event, model, time_dim):
    """Plot a histogram of the maximum duration of consecutive years that meet the threshold."""
    _, ds_max = gsr_events(
        data,
        decile,
        threshold=event.threshold,
        min_duration=2,
        operator=event.operator,
        fixed_duration=False,
        minimize=False,
        time_dim=time_dim,
    )

    if model != "AGCD":
        ds_max = ds_max.rename({"event": "ev"})
        ds_max = ds_max.stack({"event": ["init_date", "ensemble", "ev"]})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{model} duration of consecutive years with GSR {event_max.decile}")

    for i, ax in enumerate(axes):
        bins = np.arange(
            event.min_duration - 0.5, ds_max.isel(x=i).duration.max().load().item() + 1
        )
        ds_max.duration.isel(x=i).plot.hist(ax=ax, bins=bins, color="b", edgecolor="k")
        ax.set_title(f"{regions[i]} region")
        ax.set_xlabel("Duration [years]")
        ax.set_ylabel("Frequency")
        ax.set_xmargin(1e-3)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(home / f"figures/{event.event_type[:4]}_duration_{model}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    for model in models:
        for operator in ["less", "greater"]:
            event_max = Events(
                min_duration=2, operator=operator, minimize=False, fixed_duration=False
            )
            if model == "AGCD":
                data, decile = get_AGCD_data_regions(regions)
                time_dim = "time"
                events = [
                    Events(min_duration=i, operator=operator, minimize=j)
                    for i in [1, 2, 3]
                    for j in [True, False]
                ]
                for event in [event_max, *events]:
                    plot_AGCD_timeseries(data, decile, event)
            else:
                data, decile = get_DCPP_data_regions(model, regions)
                time_dim = "lead_time"
                event = Events(min_duration=3, operator=operator, minimize=True)
                try:
                    plot_DCPP_timeseries(data, decile, event, model, n=0)
                except Exception:
                    print(model, operator, "failed")

            event = Events(min_duration=3, operator=operator)
            next_year_decile_histograms(decile, model, event, time_dim, binned=False)
            next_year_decile_histograms(decile, model, event, time_dim, binned=True)

            transition_probability_matrix(decile, model, event)

            max_duration_histogram(data, decile, event_max, model, time_dim)
