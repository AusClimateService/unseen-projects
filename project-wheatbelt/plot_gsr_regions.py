"""Plots of low/high growing season (Apr-Oct) rainfall events in the WA and SA regions."""

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator
import numpy as np
import xarray as xr

from process_gsr_data import home, models, gsr_data_regions
from gsr_events import (
    gsr_events,
    transition_probability,
    downsampled_transition_probability,
    transition_time,
)
from utils import binom_ci

plt.rcParams["font.size"] = 12
regions = ["WA", "SA"]


def plot_duration_histogram(dv, event, model, time):
    """Plot a histogram of the maximum duration of consecutive years that meet the threshold."""
    _, ds_max = gsr_events(
        dv.pr,
        dv.decile,
        threshold=event.threshold,
        min_duration=2,
        operator=event.operator,
        fixed_duration=False,
        minimize=False,
        time=time,
    )
    bins = np.arange(1.5, 11)
    if model != "AGCD":
        ds_max = ds_max.rename({"event": "ev"})
        ds_max = ds_max.stack({"event": ["init_date", "ensemble", "ev"]})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(model)

    for i, ax in enumerate(axes):
        ds_max.duration.isel(x=i).plot.hist(
            ax=ax, bins=bins, color="b", edgecolor="k", align="mid"
        )
        ax.set_title(f"{regions[i]} region consecutive {event.alt_name} years")
        ax.set_xlabel("Duration [years]")
        ax.set_ylabel("Frequency")
        ax.set_xmargin(1e-3)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_duration_histogram_{model}.png", dpi=200
    )
    plt.show()


def plot_duration_histogram_downsampled(model, models, event, time, n_samples=10000):
    """Plot a histogram of the maximum duration of consecutive years that meet the threshold."""
    rng = np.random.default_rng(seed=42)
    time = "lead_time"
    min_duration = 2
    bins = np.arange(min_duration, 11)
    xmodels = models if model == "all" else [model]
    # Model whisker colors
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(models) + 1))

    dv, ds = [], []
    for m in ["AGCD", *xmodels]:
        dv_m = gsr_data_regions(m, regions)
        _, ds_m = gsr_events(
            dv_m.pr,
            dv_m.decile,
            threshold=event.threshold,
            min_duration=min_duration,
            operator=event.operator,
            fixed_duration=False,
            minimize=False,
            time=time if m != "AGCD" else "time",
        )
        if m != "AGCD":
            ds_m = ds_m.stack(dict(sample=["ensemble", "init_date"]))
        ds.append(ds_m.duration)
        dv.append(dv_m)

    # Number of sub-samples to match AGCD data (123 years)
    n_agcd = dv[0].time.size

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ax in enumerate(axes):
        ax.set_title(f"{regions[i]} region consecutive {event.alt_name} years")
        # AGCD histogram
        counts_agcd, _ = np.histogram(ds[0].isel(x=i), bins=bins)
        ax.bar(
            bins[:-1],
            (counts_agcd / n_agcd) * 100,
            color="b",
            edgecolor="k",
            width=1,
            label="AGCD",
            alpha=0.5 if model == "all" else 1,
        )
        # Multi-model boxplot
        for m in range(len(xmodels)):
            # Get n samples of subsampled data
            dx = ds[m + 1].isel(x=i, drop=True)
            target = n_agcd // dv[m + 1][time].size
            # subsample=target * lead_time ("events" were calculated over lead_time dim)
            dx_sampled = rng.choice(dx.T, (n_samples, target), replace=True, axis=0)
            dx_sampled = xr.DataArray(dx_sampled, dims=("sample", "target", "event"))
            # Stack the events in each subsample (sample, target, event) -> (sample, subsample)
            dx_sampled = dx_sampled.stack(dict(subsample=["target", "event"]))
            counts, _ = xr.apply_ufunc(
                np.histogram,
                dx_sampled,
                input_core_dims=[["subsample"]],
                output_core_dims=[["bin"], ["bin_edges"]],
                vectorize=True,
                dask="parallelized",
                kwargs=dict(bins=bins),
                output_dtypes=["int"] * ((len(bins) * 2) - 1),
            )
            # Model boxplot
            box_kwargs = dict(
                positions=(bins[:-1] - 0.5) + (m + 1) / 10,
                widths=0.03,
                boxprops=dict(lw=0.4),
                whiskerprops=dict(color=colors[m]),
                flierprops=dict(ms=2, markeredgecolor=colors[m]),
            )
            box_kwargs = box_kwargs if model == "all" else dict(positions=bins[:-1])
            ax.boxplot(
                (counts / (dv[m + 1][time].size * target)) * 100,
                whis=[5, 95],
                sym=".",
                **box_kwargs,
            )
        ax.set_xticks(bins[:-1])
        ax.set_xticklabels(bins[:-1])
        ax.set_xlabel("Duration [years]")
        ax.set_ylabel("Probability [%]")
        ax.set_xlim(bins[0] - 0.5, bins[-1] - 0.5)

        if model == "all":
            # Add model colors to the legend
            _ = [ax.plot([], [], label=m, color=c) for m, c in zip(models, colors)]

        lgd = ax.legend(loc="upper right", fontsize=11)

    if model != "all":
        fig.suptitle(
            f"AGCD ({n_agcd} years) and {model} ({n_samples} samples of {target}x{dv[1][time].size} years)"
        )
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_duration_histogram_xsamples_{model}.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()


def transition_histogram(ax, k, total, bins, var, alpha=1):
    """Plot the histogram of next year GSR terciles/deciles/binned_deciles."""
    if not isinstance(total, int):
        total = int(total.load().item())
    x = np.arange(len(bins) - 1)

    # Sample colors from the BrBG colormap
    if len(bins) == 4:
        rescale = np.array([0.08, 0.42, 0.82])
    else:
        rescale = (bins[:-1] - np.min(bins[:-1])) / np.ptp(bins[:-1])
    colors = plt.cm.BrBG(rescale)

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
    ax.set_ylabel("Probability [%]")
    ax.yaxis.labelpad = 2

    if len(bins) == 4:
        ax.set_xticklabels(["Dry", "Medium", "Wet"])
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


def plot_transition_duration_histogram(decile, model, event, time):
    """Histogram of duration between a n-year low decile event and a high decile year."""
    transition = ["dry", "wet"]
    if event.operator == "greater":
        transition = transition[::-1]

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    fig.suptitle(
        f"{model} years between n {transition[0]} years and a {transition[1]} year"
    )

    for i, region in enumerate(regions):
        for j, n in enumerate([1, 2, 3]):
            k, bins = transition_time(decile, n, time, transition[0])

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
            ax[i, j].set_title(f"{region} region after {n}-year event", loc="left")
            ax[i, j].set_xlabel("Duration [years]")
            ax[i, j].set_ylabel("Frequency")
            ax[i, j].set_xmargin(1e-3)
            ax[i, j].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax[i, j].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_duration_histogram_{model}.png",
        dpi=200,
    )
    plt.show()


def plot_transition_histogram(da, model, event, time="time", var="decile"):
    """Plot the histogram of next year GSR deciles after n years."""
    # Calculate the transition counts & event totals
    ds = xr.Dataset()
    ds["k"], ds["total"], bins = transition_probability(
        da,
        event.threshold,
        event.operator,
        min_duration=np.arange(1, 4),
        var=var,
        time=time,
        binned=True,
    )
    # Sum over other dimensions
    dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
    ds = ds.sum(dims)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    title = f"{model} {event.name} {var.replace('binned_', '')} GSR events"
    if var == "decile":
        title += f" (Apr-Oct rain {event.decile})"
    plt.suptitle(title)

    for j, N in enumerate([1, 2, 3]):
        for i, region in enumerate(regions):
            ax[i, j].set_title(f"{region} region GSR after {N}yr event")
            dx = ds.isel(x=i, n=j)

            ax[i, j] = transition_histogram(ax[i, j], dx.k, dx.total, bins, var)

    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_histogram_{var}_{model}.png",
        dpi=200,
    )
    plt.show()


def plot_transition_histogram_downsampled(tercile, model, event, n_samples=1000):
    """Plot next year GSR tercile histograms (AGCD with downsampled model box & whiskers)."""

    ds = downsampled_transition_probability(
        tercile, event, regions, target="AGCD", n_samples=n_samples
    )
    bins = np.arange(1, 5)

    dims = [d for d in ds.k.dims if d not in ["n", "x", "sample", "q"]]
    ds = ds.sum(dims)

    _, ax = plt.subplots(2, 3, figsize=(12, 7))
    plt.suptitle(
        f"AGCD {event.name} tercile GSR events ({model} random samples={n_samples})"
    )
    for j, N in enumerate([1, 2, 3]):
        for i, r in enumerate(regions):
            ax[i, j].set_title(f"{r} region GSR after {N}yr event", loc="left", x=-0.01)
            dx = ds.isel(x=i, n=j)

            # AGCD histogram bars (decrease bar transparency)
            ax[i, j] = transition_histogram(
                ax[i, j], dx.k_agcd, dx.total, bins, "tercile", alpha=0.7
            )

            # Model boxplot
            ax[i, j].boxplot(
                (dx.k.T / dx.total) * 100, whis=[5, 95], positions=[0.5, 1.5, 2.5]
            )
            ax[i, j].set_xticks([0.5, 1.5, 2.5])
            ax[i, j].set_xticklabels(["Dry", "Medium", "Wet"])
            ax[i, j].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_histogram_xsamples_{model}.png",
        dpi=200,
    )
    plt.show()


def plot_transition_histogram_downsampled_all_models(event, n_samples=1000):
    """Plot next year GSR tercile histograms (AGCD with downsampled model box & whiskers)."""
    dss = []
    for model in models:
        dv = gsr_data_regions(model, regions)
        ds = downsampled_transition_probability(
            dv.tercile, event, regions, target="AGCD", n_samples=n_samples
        )
        dss.append(ds.assign_coords(model=model))
    ds = xr.concat(dss, dim="model")
    ds["k_agcd"] = ds.k_agcd.isel(model=0, drop=True)
    ds["total"] = ds.total.isel(model=0, drop=True)

    bins = np.arange(1, 5)

    dims = [d for d in ds.k.dims if d not in ["n", "x", "sample", "q", "model"]]
    ds = ds.sum(dims)
    # Model whisker colors
    colors = mpl.colormaps.get_cmap("rainbow")(np.linspace(0, 1, len(models) + 1))

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    plt.suptitle(f"AGCD {event.name} tercile GSR events (random samples={n_samples})")
    for j, N in enumerate([1, 2, 3]):
        for i, r in enumerate(regions):
            ax[i, j].set_title(
                f"{r} region GSR after {N}yr event", loc="left", x=-0.025
            )
            dx = ds.isel(x=i, n=j)

            # AGCD histogram bars (decrease bar transparency)
            ax[i, j] = transition_histogram(
                ax[i, j], dx.k_agcd, dx.total, bins, "tercile", alpha=0.7
            )

            # Model boxplot
            for m in range(len(models)):

                ax[i, j].boxplot(
                    dx.k.isel(model=m).T,
                    whis=[5, 95],
                    positions=np.arange(1, 4) + (m + 0.6) / 10,
                    widths=0.03,
                    sym=".",
                    boxprops=dict(lw=0.4),
                    whiskerprops=dict(color=colors[m]),
                    flierprops=dict(ms=2, markeredgecolor=colors[m]),
                )
            ax[i, j].set_xticks([1.5, 2.5, 3.5])
            ax[i, j].set_xticklabels(["Dry", "Medium", "Wet"])
            ax[i, j].set_xlim(1, 4)
    lines = [
        mpl.lines.Line2D([0], [0], label=m, color=c) for m, c in zip(models, colors)
    ]
    lgd = fig.legend(
        handles=lines,
        loc="upper left",
        bbox_to_anchor=(1, 0.9),
        fontsize=12,
        title="Model",
    )
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_histogram_xsamples_all_models.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        dpi=200,
    )
    plt.show()


def plot_transition_matrix_downsampled(da, model, event):
    """Plot the histogram of next year GSR terciles after n years with resampling."""
    # Get samples of the downsampled transition matrices
    nr, nc = 5, 5
    n_samples = nr * nc

    ds = downsampled_transition_probability(
        da, event, regions, target="AGCD", n_samples=n_samples
    )

    dims = [d for d in ds.k.dims if d not in ["n", "q", "x", "sample"]]
    ds = ds.sum(dims)
    ds["p"] = (ds.k / ds.total) * 100

    # Calculate the 95% confidence interval
    ci0, ci1 = binom_ci(ds.total, p=1 / 3)
    ds["mask"] = ds.p.where((ds.k < ci0) | (ds.k > ci1))

    x = np.arange(3)
    cmap = cmocean.tools.crop_by_percent(plt.cm.PuOr, 60 - 100 / 3, which="min", N=None)
    levels = FixedLocator(np.arange(0, 1, 1 / 12) * 100).tick_values(0, 90)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Plot nr samples of transition histograms at a region
    for i, r in enumerate(regions):

        fig, axes = plt.subplots(nr, nc, figsize=(16, 14), sharey=True)
        axes = axes.flatten()
        plt.suptitle(
            f"{r} region: {model} {event.name} tercile GSR event probability of transition [%] (same event sample sizes as AGCD)"
        )
        for s in range(nr * nc):
            ax = axes[s]
            cm = ax.pcolormesh(ds.p.isel(x=i, sample=s), cmap=cmap, norm=norm)
            ax.set_xticks(x + 0.5)
            ax.set_yticks((x + 0.5))
            ax.set_yticklabels((x + 1))
            ax.set_xticklabels(["Dry", "Medium", "Wet"])

            ax.set_title(f"Random sample #{s + 1}", fontsize=11, loc="left")

            ax.pcolor(
                ds.mask.isel(x=i, sample=s),
                cmap=mpl.colors.ListedColormap(["none"]),
                hatch=".",
                ec="k",
                zorder=1,
                lw=0,
            )
            if s >= nc * (nr - 1):
                # Add xlabel to the bottom of each column
                ax.set_xlabel("Next year Apr-Oct rainfall", fontsize=11)
            if s % nc == 0:
                # At label at the start of each row
                ax.set_ylabel(f"Consecutive {event.name} tercile years", fontsize=11)
            cbar = plt.colorbar(
                cm,
                ax=ax,
                shrink=0.95,
                extend="max",
                format="%3.1f",
            )
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.savefig(
            home
            / f"figures/{event.type[:4]}_transition_matrix_xsamples_{r}_{model}.png",
            dpi=200,
        )
        plt.show()


def plot_transition_pie_chart(tercile, model, event, time="time"):
    """Plot the histogram of next year GSR deciles after n years."""
    # Calculate the transition counts & event totals
    ds = xr.Dataset()
    ds["k"], ds["total"], _ = transition_probability(
        tercile,
        event.threshold,
        event.operator,
        np.arange(1, 4),
        var="tercile",
        time=time,
    )

    # Sum over other dimensions
    dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
    ds = ds.sum(dims)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    plt.suptitle(
        f"{model} transition probability after {event.name} tercile GSR events"
    )
    for i, region in enumerate(regions):
        for j, N in enumerate([1, 2, 3]):
            ax[i, j].set_title(f"{region} region after {N}yr event")
            wedges, _, _ = ax[i, j].pie(
                ds.isel(x=i, n=j).k,
                colors=plt.cm.BrBG(np.array([0.08, 0.42, 0.82])),
                startangle=90,
                shadow=True,
                autopct="%1.1f%%",
                wedgeprops={"edgecolor": "k"},
            )
        # Add legends at the ends of the rows
        ax[i, 2].legend(
            wedges,
            ["Dry ", "Medium", "Wet"],
            title="Next year tercile",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
    plt.tight_layout()
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_pie_{model}.png",
        dpi=200,
    )
    plt.show()


def plot_transition_probability_matrix(tercile, model, event, time):
    """Plot the transition matrix for the next year GSR decile."""
    ds = xr.Dataset()
    ds["k"], ds["total"], _ = transition_probability(
        tercile,
        event.threshold,
        event.operator,
        np.arange(1, 4),
        var="tercile",
        time=time,
    )
    # Sum over other dimensions
    dims = [d for d in ds.k.dims if d not in ["n", "q", "x"]]
    ds = ds.sum(dims)
    ds["p"] = (ds.k / ds.total) * 100

    # Calculate the 95% confidence interval
    ci0, ci1 = binom_ci(ds.total, p=1 / 3)
    ds["mask"] = ds.p.where((ds.k < ci0) | (ds.k > ci1))

    x = np.arange(3)
    cmap = cmocean.tools.crop_by_percent(plt.cm.PuOr, 60 - 100 / 3, which="min", N=None)
    levels = FixedLocator(np.arange(0, 1, 1 / 12) * 100).tick_values(0, 90)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True, sharey=True
    )
    for i, region in enumerate(regions):
        ax = axes[i]
        ax.set_title(f"{model} {region} region transitions")

        # Plot probability heatmap
        cm = ax.pcolormesh(ds.p.isel(x=i), cmap=cmap, norm=norm)

        # Add hatching where significant
        ax.pcolor(
            ds.mask.isel(x=i),
            cmap=mpl.colors.ListedColormap(["none"]),
            hatch=".",
            ec="k",
            zorder=1,
            lw=0,
        )
        ax.set_yticks(x + 0.5)
        ax.set_yticklabels(x + 1)
        ax.set_xticks(x + 0.5)
        ax.set_xticklabels(["Dry", "Medium", "Wet"])
        ax.set_xlabel("Next year Apr-Oct rainfall")

    axes[0].set_ylabel(f"Consecutive {event.name} tercile years")
    fig.colorbar(
        cm,
        ax=[ax],
        label="Probability [%]",
        shrink=0.95,
        extend="max",
        format="%3.1f",
    )
    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_matrix_{model}.png", dpi=200
    )
    plt.show()


def plot_transition_sample_size(tercile, model, event, n_samples=10000):
    """Transition probabilities with different sample sizes."""
    colors = plt.cm.BrBG(np.array([0.08, 0.42, 0.82]))
    targets = 10 ** np.arange(4)
    targets[0] = 5
    xticklabels = [r"$10^{}$".format(i) for i, _ in enumerate(targets)]
    xticklabels[0] = str(5)

    fig, ax = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(f"{model} {event.name} tercile Apr-Oct rainfall events")
    for xi, x in enumerate(regions):  # WA or SA region
        for k, s in enumerate(targets):  # Sample size
            ds = downsampled_transition_probability(
                tercile, event, regions, target=s, n_samples=n_samples
            )
            for ni, n in enumerate([1, 2, 3]):  # N year events
                ax[ni, xi].set_title(
                    f"{x} region transitions after {n}yr {event.name} GSR event",
                    loc="left",
                    x=-0.03,
                )
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
                ax[ni, xi].set_xlabel("Event sample size")
                ax[ni, xi].set_ylabel("Probability [%]")
                ax[ni, xi].legend(
                    bp["boxes"], ["Dry", "Medium", "Wet"], loc="upper right"
                )

    plt.savefig(
        home / f"figures/{event.type[:4]}_transition_sample_size_{model}.png",
        dpi=200,
    )
    plt.show()


def plot_timeseries_AGCD(dv, event):
    """Plot the AGCD (rainfall & decile) timeseries with GSR events shaded."""
    events, _ = gsr_events(
        dv.pr,
        dv.decile,
        threshold=event.threshold,
        min_duration=event.n,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        minimize=event.minimize,
        time="time",
    )
    # Plot GSR rainfall or GSR decile
    for da, ylabel, fname in zip(
        [dv.pr, dv.decile], ["Rainfall [mm]", "GSR decile"], ["", "_decile"]
    ):
        _, axes = plt.subplots(2, 1, figsize=(10, 5))
        for i, state in enumerate(["Western Australia", "South Australia"]):
            axes[i].set_title(
                f"AGCD Apr-Oct rainfall in the {state} region", loc="left"
            )

        for j, ax in enumerate(axes):
            # Plot timeseries as bars
            ax.bar(
                dv.pr.time.dt.year, da.isel(x=j), color="blue", label="Rainfall Data"
            )

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
            / f"figures/timeseries_{event.type[:4]}_AGCD{event.type[4:]}_{event.n}yr{fname}.png",
            dpi=200,
        )
        plt.show()


def plot_timeseries_DCPP(dv, event, model):
    """Plot the DCPP decile timeseries with GSR events shaded."""
    # Get the GSR events
    events, _ = gsr_events(
        dv.pr,
        dv.decile,
        threshold=event.threshold,
        min_duration=event.n,
        operator=event.operator,
        fixed_duration=event.fixed_duration,
        minimize=event.minimize,
        time="lead_time",
    )

    # Plot init_dates along columns and ensemble members along rows
    nr = dv.ensemble.size
    init_dates = np.arange(dv.init_date.size)
    nc = len(init_dates)
    year0 = dv.decile.time.dt.year.min().item()

    # Create a plot for each region
    for x, region in enumerate(regions):  # WA or SA region
        fig, ax = plt.subplots(
            nr, nc, figsize=(1.8 * nc, 1.2 * nr), sharey=True, sharex="col"
        )
        fig.suptitle(
            f"{model} {region} region {event.n}yr events of GSR {event.decile}",
        )

        # Iterate through ensemble members
        for j in range(nr):
            # Iterate through init_dates
            for i, init_date in enumerate(init_dates[:nc]):
                loc = {"ensemble": j, "init_date": init_date, "x": x}
                da = dv.decile.isel(loc)
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

            ax[j, 0].set_ylabel("Decile")

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
            home
            / f"figures/timeseries_{event.type[:4]}_{model}{event.type[4:]}_{region}_{event.n}yr.png",
            dpi=200,
        )
        plt.show()
