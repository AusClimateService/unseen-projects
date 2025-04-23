# -*- coding: utf-8 -*-
"""Multi-model UNSEEN spatial maps using CAFE and DCPP models.

Notes
-----
* Slices obs_ds time period to match models
* requires xarray>=2024.10.0, numpy<=2.1.0 (shapely issue?)
* requires acs_plotting_maps

"""

import argparse
import calendar

# import cartopy.feature as cfeature
from cartopy.crs import PlateCarree
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
import functools

# import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
from pathlib import Path
from scipy.stats import mode
import string
import time
import xarray as xr

from unseen import fileio, time_utils, eva, general_utils
from unseen.stability import statistic_by_lead_confidence_interval


from acs_plotting_maps import cmap_dict, tick_dict, regions_dict  # noqa
from spatial_plots import (
    InfoSet,
    func_dict,
    soft_record_metric,
    resample_subsample,
    nonstationary_new_record_probability,
    month_cmap,
    new_record_probability_empirical,
)

plt.rcParams["figure.figsize"] = [14, 10]
plt.rcParams["figure.dpi"] = 600
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["contour.linewidth"] = 0.3

# Subplot letter labels
letters = [f"({i})" for i in string.ascii_letters]

models = np.array(
    [
        "CAFE",
        "BCC-CSM2-MR",
        "CanESM5",
        "CMCC-CM2-SR5",
        "EC-Earth3",
        "HadGEM3-GC31-MM",
        "IPSL-CM6A-LR",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "MRI-ESM2-0",
        "NorCPM1",
    ]
)


# ----------------------------------------------------------------------------
# Generic plotting functions
# ----------------------------------------------------------------------------


def map_subplot(
    fig,
    ax,
    data,
    region="aus_states_territories",
    stippling=None,
    title=None,
    ticks=None,
    ticklabels=None,
    cmap=plt.cm.viridis,
    norm=None,
    cbar=None,
    cbar_label=None,
    cbar_extend="neither",
    cbar_kwargs=dict(fraction=0.05),
    mask_not_australia=True,
    xlim=(112.5, 154.3),
    ylim=(-44.5, -9.8),
    xticks=np.arange(120, 155, 10),
    yticks=np.arange(-40, -0, 10),
):
    """Plot 2D data on an Australia map with coastlines.

    Returns
    -------
    fig : matplotlib figure
    ax : cartopy.mpl.geocollection.GeoQuadMesh
    cs : cartopy.mpl.geocollection.GeoQuadMesh

    Example
    -------
    data = xr.DataArray(
        np.random.rand(10, 10),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-45, -10, 10), "lon": np.linspace(115, 155, 10)},
        )
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), subplot_kw=dict(projection=PlateCarree()))
    fig, ax, cs = map_subplot(fig, ax, data, title="Random data", cmap="viridis")
    """

    if title is not None:
        ax.set_title(title, loc="left", fontsize=12)

    ax.set_extent([*xlim, *ylim], crs=PlateCarree())
    middle_ticks = ticks
    if ticks is not None:
        if isinstance(ticks, int):
            # Set number of ticks between min and max values
            ticks = MaxNLocator(nbins=ticks).tick_values(data.min(), data.max())

        elif isinstance(ticks, (list, np.ndarray)):
            # Stolen from acs_plotting_maps
            if ticklabels is None or (len(ticklabels) == len(ticks) - 1):
                norm = BoundaryNorm(ticks, cmap.N + 1, extend=cbar_extend)
                if ticklabels is not None:
                    middle_ticks = [
                        (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
                    ]
                else:
                    middle_ticks = []
            else:
                middle_ticks = [
                    (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
                ]
                outside_bound_first = [ticks[0] - (ticks[1] - ticks[0]) / 2]
                outside_bound_last = [ticks[-1] + (ticks[-1] - ticks[-2]) / 2]
                bounds = outside_bound_first + middle_ticks + outside_bound_last
                norm = BoundaryNorm(bounds, cmap.N, extend="neither")

    cs = ax.pcolormesh(data.lon, data.lat, data, cmap=cmap, norm=norm)

    if stippling is not None:
        ax = add_stippling(ax, stippling)

    if mask_not_australia:
        ax = plot_region_mask(ax, regions_dict["not_australia"])

    if cbar not in [None, False]:
        fig.colorbar(
            cs, extend=cbar_extend, label=cbar_label, ticks=ticks, **cbar_kwargs
        )

    if region is not None:
        ax = plot_region_border(ax, regions_dict[region], ec="k", zorder=5)

    # Format axis ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fig, ax, cs


def add_stippling(ax, stippling, **kwargs):
    """Add hatching to contour plot."""
    ax.contourf(
        stippling.lon,
        stippling.lat,
        stippling,
        alpha=0,
        hatches=["", "xxxxx"],
        transform=PlateCarree(),
        **kwargs,
    )
    return ax


def add_shared_colorbar(
    fig,
    ax,
    cs,
    orientation="horizontal",
    ticks=None,
    ticklabels=None,
    hide_mid_ticklabels=None,
    **kwargs,
):
    """Add a shared colorbar to a figure with multiple subplots."""

    # Set default colorbar parameters
    kwargs["aspect"] = 32
    if "pad" not in kwargs and "shrink" not in kwargs:
        if orientation == "vertical":
            kwargs["pad"] = 0.03
            kwargs["shrink"] = 0.75
        else:
            kwargs["pad"] = 0.02
            kwargs["shrink"] = 0.7

    # Format ticks
    if ticks is not None:
        if isinstance(ticks, (list, np.ndarray)):
            # Stolen from acs_plotting_maps
            if ticklabels is None or (len(ticklabels) == len(ticks) - 1):
                if ticklabels is not None:
                    middle_ticks = [
                        (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
                    ]
                else:
                    middle_ticks = []
            else:
                middle_ticks = [
                    (ticks[i + 1] + ticks[i]) / 2 for i in range(len(ticks) - 1)
                ]

    cbar = fig.colorbar(cs, ax=ax, orientation=orientation, ticks=ticks, **kwargs)

    if ticklabels is not None:
        if orientation == "vertical":
            cbar.ax.set_yticks(middle_ticks, ticklabels)
        else:
            cbar.ax.set_xticks(middle_ticks, ticklabels)

    if hide_mid_ticklabels is not None:
        # Hide every other tick label
        if orientation == "vertical":
            labels = cbar.ax.yaxis.get_ticklabels()
        else:
            labels = cbar.ax.xaxis.get_ticklabels()
        for label in labels[::2]:
            label.set_visible(False)
    return cbar


def plot_region_border(ax, region, lw=0.5, ec="k", **kwargs):
    """Mask data outside of the region."""
    ax.add_geometries(
        region.geometry,
        linewidth=lw,
        ls="-",
        facecolor="none",
        edgecolor=ec,
        crs=PlateCarree(),
        **kwargs,
    )
    return ax


def plot_region_mask(ax, region, facecolor="white", **kwargs):
    """Mask data outside of the region."""
    ax.add_geometries(
        region,
        crs=PlateCarree(),
        linewidth=0,
        facecolor=facecolor,
        **kwargs,
    )
    return ax


# ----------------------------------------------------------------------------
# Data loading and processing functions
# ----------------------------------------------------------------------------

def get_makefile_vars(
    metric="txx", obs="AGCD", obs_config_file="AGCD-CSIRO_r05_tasmax_config.mk"
):
    """Create nested dictionary of observation and model variables used in Makefile.

    Saves all observation and model local variables defined in the Makefile (converted
    to lower case and sorted) for the given metric and observation dataset.

    Parameters
    ----------
    metric : str
        Metric to plot (used to find project details).
    obs : str
        Name of the observation dataset
    obs_config_file : str
        Name of the observation dataset makefile (without path).

    Returns
    -------
    var_dict : dict
        Nested dictionary of e.g., {CAFE: {metric_fcst: filename, metric_obs: filename}}
    """

    dir = Path("/g/data/xv83/unseen-projects/code/")
    obs_details = dir / f"dataset_makefiles/{obs_config_file}"
    project_details = dir / f"project-{metric}/{metric}_config.mk"

    # Nested dictionary of {model: {key1: value, key2: value}}
    var_dict = {}
    var_dict[obs] = general_utils.get_model_makefile_dict(
        dir, project_details, obs, obs_details, obs_details
    )

    # Format/eval dictionary values (observation only?)
    for key in ["plot_dict", "gev_trend_period"]:
        var_dict[obs][key] = eval(var_dict[obs][key])
    var_dict[obs]["reference_time_period"] = list(
        var_dict[obs]["reference_time_period"].split(" ")
    )

    for model in models:
        # Search for the model makefile
        model_details = list((dir / "dataset_makefiles/").rglob(f"{model}*config.mk"))[
            0
        ]
        var_dict[model] = general_utils.get_model_makefile_dict(
            dir, project_details, model, model_details, obs_details
        )
    return var_dict


def open_model_dataset(
    kws,
    bc=None,
    alpha=0.05,
    time_dim="time",
    init_dim="init_date",
    lead_dim="lead_time",
    ensemble_dim="ensemble",
):
    """Open, format and combine relevant model data."""

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    var = kws["var"]
    if bc not in ["", None]:
        _bc = f"_{bc}_bc"
    else:
        _bc = ""
    metric_fcst = kws[f"metric_fcst{_bc}"]
    similarity_file = str(kws[f"similarity{_bc}_file"])
    gev_params_nonstationary_file = kws[f"gev_nonstationary{_bc}"]

    model_ds = fileio.open_dataset(metric_fcst)

    # Similarity test (for stippling)
    similarity_ds = fileio.open_dataset(similarity_file)
    ds_independence = xr.open_dataset(kws["independence_file"], decode_times=time_coder)
    # Non-stationary GEV parameters
    dparams_ns = xr.open_dataset(gev_params_nonstationary_file)[var]

    # Calculate stability confidence intervals (for median and 1% AEP)
    if bc is not None:
        da = fileio.open_dataset(kws[f"metric_fcst"])[var]
    else:
        da = model_ds[var]
    ci_median = get_stability_ci(da, "median", confidence_level=0.99, n_resamples=1000)
    ci_aep = get_stability_ci(da, "aep", confidence_level=0.99, n_resamples=1000, aep=1)

    min_lead_ds = fileio.open_dataset(
        kws["independence_file"],
        variables="min_lead",
        shapefile=kws["shapefile"],
        shape_overlap=kws["shape_overlap"],
        spatial_agg=kws["min_lead_shape_spatial_agg"],
    )

    # Drop dependent lead times
    min_lead = min_lead_ds["min_lead"]
    model_ds = model_ds.groupby(f"{init_dim}.month").where(
        model_ds[lead_dim] >= min_lead
    )
    model_ds = model_ds.dropna(lead_dim, how="all")
    model_ds = model_ds.sel(lat=dparams_ns.lat, lon=dparams_ns.lon)

    # Convert event_time from string to cftime
    try:
        event_times = np.vectorize(time_utils.str_to_cftime)(
            model_ds.event_time, model_ds.time.dt.calendar
        )
        model_ds["event_time"] = (model_ds.event_time.dims, event_times)
    except ValueError:
        model_ds["event_time"] = model_ds.event_time.astype(
            dtype="datetime64[ns]"
        ).compute()

    model_ds_stacked = model_ds.stack(
        {"sample": [ensemble_dim, init_dim, lead_dim]}, create_index=False
    )
    model_ds_stacked = model_ds_stacked.transpose("sample", ...)
    model_ds_stacked = model_ds_stacked.dropna("sample", how="all")
    model_ds_stacked = model_ds_stacked.chunk(dict(sample=-1))

    model_ds_stacked[f"{kws['similarity_test']}_pval"] = similarity_ds[
        f"{kws['similarity_test']}_pval"
    ]
    model_ds_stacked["pval_mask"] = (
        similarity_ds[f"{kws['similarity_test']}_pval"] <= alpha
    )
    model_ds_stacked["dparams_ns"] = dparams_ns
    model_ds_stacked["covariate"] = model_ds_stacked[time_dim].dt.year

    for dvar in ds_independence.data_vars:
        model_ds_stacked[dvar] = ds_independence[dvar]
    model_ds_stacked["min_lead_median"] = min_lead
    model_ds_stacked["ci_median"] = ci_median
    model_ds_stacked["ci_aep"] = ci_aep

    return model_ds_stacked


def open_obs_dataset(var_dict, obs):
    """Open metric observational datasets."""
    obs_ds = fileio.open_dataset(var_dict[obs]["metric_obs"])
    ns_gev_file = Path(var_dict[obs]["gev_nonstationary_obs"])
    dparams_ns = xr.open_dataset(ns_gev_file)[var_dict[obs]["var"]]

    if var_dict[obs]["reference_time_period"] is not None:
        obs_ds = time_utils.select_time_period(
            obs_ds, var_dict[obs]["reference_time_period"]
        )

    event_times = np.vectorize(time_utils.str_to_cftime)(
        obs_ds.event_time, obs_ds.time.dt.calendar
    )
    obs_ds["event_time"] = (obs_ds.event_time.dims, event_times)
    obs_ds["dparams_ns"] = dparams_ns
    return obs_ds


def subset_obs_dataset(obs_ds, ds):
    """Subset observation dataset to the given time period."""
    start_year = ds.time.dt.year.min().load().item()
    obs_ds = obs_ds.where(obs_ds.time.dt.year >= start_year, drop=True)
    obs_ds = obs_ds.dropna("time", how="all")
    return obs_ds

# def multimodel_median(da_list):
#     # Regrid to commonn 1x1 degree grid
#     # create a common grid
#     common_grid = xr.Dataset(

def get_stability_ci(da, method, confidence_level=0.99, n_resamples=1000, aep=1):
    """Get the confidence interval of a statistic for a lead time-sized sample.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray of the model data.
    method : {"median", "aep"}
        Method to use for the statistic.
    confidence_level : float, default=0.99
        Confidence level for the confidence interval.
    n_resamples : int, default=1000
        Number of resamples to use for the confidence interval.
    aep : float, default=1
        Annual exceedance probability (for the AEP method).

    Returns
    -------
    ci : xarray.DataArray
        Confidence interval of the statistic.
    """

    if method == "median":
        statistic = np.median
        kwargs = {}

    elif method == "aep":
        # Pass the return period to the statistic function
        statistic = eva.empirical_return_level
        kwargs = dict(return_period=eva.aep_to_ari(aep))

    ci = statistic_by_lead_confidence_interval(
        da,
        statistic,
        sample_size=da.ensemble.size * da.init_date.size,
        method="percentile",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=np.random.default_rng(0),
        **kwargs,
    )
    return ci


# ----------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------


def plot_time_agg(info, var, time_agg):
    """Plot time-aggregated data for each model and observation dataset."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        da = info[m].ds[var].reduce(func_dict[time_agg], dim=info[m].time_dim)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap,
            ticks=info[m].ticks,
            cbar_extend=info[m].cbar_extend,
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=info[m].units_label,
        ticks=info[m].ticks,
        extend=info[m].cbar_extend,
    )

    suptitle = f"{time_agg.capitalize()} {info[m].metric}"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/{time_agg}_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_time_agg_subsampled(info, obs, time_agg="maximum", resamples=1000):
    """Plot map of observation-sized subsample of data (sample median of time-aggregate)."""

    n_obs_samples = info[obs].obs_ds[info[obs].var].time.size

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    # Show obs time agg (not subsampled)
    i = 0
    da_obs = info[obs].ds[var].reduce(func_dict[time_agg], dim=info[obs].time_dim)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        da_obs,
        title=f"{letters[i]} {info[obs].name} {info[obs].metric} {time_agg}",
        cbar=False,
        stippling=info[obs].pval_mask,
        cmap=info[obs].cmap,
        ticks=info[obs].ticks,
        cbar_extend=info[obs].cbar_extend,
    )

    da_list = []
    for i, m in enumerate(models):
        i += 1
        da = resample_subsample(info[m], info[m].ds, time_agg, n_obs_samples, resamples)
        da_list.append(da)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap,
            ticks=info[m].ticks,
            cbar_extend=info[m].cbar_extend,
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=info[m].units_label,
        ticks=info[m].ticks,
        extend=info[m].cbar_extend,
    )

    suptitle = f"{info[m].metric} {time_agg} in obs-sized subsample (median of {resamples} resamples)"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/{time_agg}_subsampled_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()

    # Plot anomaly of subsampled data minus regrided obs data
    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    for i, m in enumerate(models):
        da = da_list[i]
        da_obs_regrid = general_utils.regrid(da_obs, da)
        da = da - da_obs_regrid
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap_anom,
            ticks=info[m].ticks_anom,
            cbar_extend=info[m].cbar_extend,
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=info[m].units_label,
        ticks=info[m].ticks_anom,
        extend=info[m].cbar_extend,
    )

    suptitle = f"{info[m].metric} {time_agg} in obs-sized subsample (median of {resamples} resamples; observed anomaly)"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/{time_agg}_subsampled_anom_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_obs_anom(
    info, var, time_agg="maximum", metric="anom", covariate_base=2025, ticks=None
):
    """Plot map of soft-record metric (e.g., anomaly) between model and observation."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    for i, m in enumerate(models):
        anom, kwargs = soft_record_metric(
            info[m],
            info[m].ds[var],
            info[m].obs_ds[var],
            time_agg,
            metric,
            info[m].ds["dparams_ns"],
            covariate_base,
        )
        if ticks is not None:
            kwargs["ticks"] = ticks

        fig, ax[i + 1], cs = map_subplot(
            fig,
            ax[i + 1],
            anom,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=kwargs["cmap"],
            ticks=kwargs["ticks"],
            cbar_extend=kwargs["cbar_extend"],
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    kwargs["cbar_label"] = kwargs["cbar_label"].replace("\n", " ")
    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=kwargs["cbar_label"],
        ticks=kwargs["ticks"],
        extend=kwargs["cbar_extend"],
        hide_mid_ticklabels=True if metric == "anom_pct" else None,
    )

    suptitle = kwargs["title"].replace("\n", " ")
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/{time_agg}_{metric}_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_event_month_mode(info):
    """Plot map of the most common month when event occurs."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        da = xr.DataArray(
            mode(info[m].ds.event_time.dt.month, axis=0).mode,
            coords=dict(lat=info[m].ds.lat, lon=info[m].ds.lon),
            dims=["lat", "lon"],
        )
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=month_cmap,
            ticks=np.arange(0.5, 13.5),
            ticklabels=list(calendar.month_name)[1:],
            cbar_extend="neither",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        extend="neither",
        ticks=np.arange(0.5, 13.5),
        ticklabels=list(calendar.month_abbr)[1:],
    )

    suptitle = f"{info[m].metric} most common month"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/month_mode_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_event_year(info, var, time_agg="maximum", ticks=np.arange(1960, 2025, 5)):
    """Plot map of the year of the maximum or minimum event."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        dt = info[m].ds[var].copy().compute()
        dt.coords[info[m].time_dim] = dt.event_time.dt.year

        if time_agg == "maximum":
            da = dt.idxmax(info[m].time_dim)
        elif time_agg == "minimum":
            da = dt.idxmin(info[m].time_dim)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            # stippling=info[m].pval_mask,
            cmap=cmap_dict["inferno"],
            ticks=ticks,
            cbar_extend="max",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label="",
        extend="max",
        ticks=ticks,
        ticklabels=None,
    )

    suptitle = f"Year of {time_agg} {info[m].metric}"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/year_{time_agg}_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_nonstationary_gev_param(info, param="loc1"):
    """Plot map of GEV location and scale parameter trends."""
    m = list(info.keys())[0]
    param_dict_dict = {
        "c": dict(
            name="shape", units="", ticks=np.arange(-1, 1.1, 0.2), cmap=plt.cm.RdBu_r
        ),
        "loc0": dict(
            name="location intercept",
            units="",
            ticks=np.arange(-150, 170, 20),
            cmap=plt.cm.RdBu_r,
        ),
        "loc1": dict(
            name="location trend",
            units=" / decade",
            ticks=info[m].ticks_param_trend["location"],
            cmap=plt.cm.RdBu_r,
        ),
        "scale0": dict(
            name="scale intercept",
            units="",
            ticks=np.arange(0, 32, 2),
            cmap=plt.cm.viridis,
            extend="max",
        ),
        "scale1": dict(
            name="scale trend",
            units=" / decade",
            ticks=info[m].ticks_param_trend["scale"],
            cmap=plt.cm.RdBu_r,
        ),
    }
    assert param in param_dict_dict.keys(), f"Unknown parameter: {param}"
    param_dict = param_dict_dict[param]

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        da = info[m].ds.dparams_ns.sel(dparams=param)
        if param in ["loc1", "scale1"]:
            da = da * 10  # Convert to per decade
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=param_dict["cmap"],
            ticks=param_dict["ticks"],
            cbar_extend=param_dict["extend"] if "extend" in param_dict else "both",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"{param_dict['name'].capitalize()} parameter {param_dict['units']}",
        ticks=param_dict["ticks"],
        extend=param_dict["extend"] if "extend" in param_dict else "both",
    )

    suptitle = f"{info[m].metric} GEV distribution {param_dict['name']} parameter"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/gev_{param_dict['name']}_{info[m].filestem}.png"
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile, bbox_inches="tight")


def plot_min_independent_lead(info):
    """Plot map of the minimum independent lead time for each model.

    The minimum independent lead time is the first lead time in which the
    ensemble mean correlation coefficient is within the 99% confidence interval.
    """

    ticks = np.arange(0, 11)
    ticklabels = np.arange(1, 11)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    i = 0
    for m in models:
        da = info[m].ds
        for month in da.month.values:
            dx = da.min_lead.sel(month=month)
            fig, ax[i + 1], cs = map_subplot(
                fig,
                ax[i + 1],
                dx,
                title=f"{letters[i]} {info[m].name} ({calendar.month_name[month]} starts)",
                cbar=False,
                cmap=plt.cm.viridis,
                ticks=ticks,
                ticklabels=ticklabels,
                cbar_extend="neither",
            )

            # Add value of min lead spatial median in lower left corner with border
            min_lead_median = da.min_lead_median.load()
            if "month" in min_lead_median.dims:
                min_lead_median = min_lead_median.sel(month=month)

            ax[i + 1].text(
                0.05,
                0.05,
                f"median={min_lead_median.item() + 1:.0f}",
                ha="left",
                va="bottom",
                transform=ax[i + 1].transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                    facecolor="white",
                    alpha=0.8,
                ),
            )
            i += 1

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label="First independent lead time",
        extend="neither",
        ticks=ticks,
        ticklabels=ticklabels,
    )

    suptitle = f"First independent {info[m].metric} lead time"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/independence_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_aep(info, covariate, aep=1):
    """Plot maps of AEP for a given threshold (at a covariate value)."""

    ari = eva.aep_to_ari(aep)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        # _covariate = xr.DataArray([covariate], dims=info[m].time_dim)
        da = eva.get_return_level(ari, info[m].ds.dparams_ns, covariate)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap,
            ticks=info[m].ticks,
            cbar_extend=info[m].cbar_extend,
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=info[m].units_label,
        ticks=info[m].ticks,
        extend=info[m].cbar_extend,
    )
    suptitle = f"{info[m].metric} {aep}% annual exceedance probability"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/aep_{aep:g}pct_{covariate:.0f}_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_aep_trend(info, covariates, aep=1, ticks=None):
    """Plot map of the change in AEP between two covariate (list) values."""

    ari = eva.aep_to_ari(aep)
    ticks = info[models[0]].ticks_anom if ticks is None else ticks

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        _covariates = xr.DataArray(covariates, dims=info[m].time_dim)
        da = eva.get_return_level(ari, info[m].ds.dparams_ns, _covariates)
        da = da.isel({info[m].time_dim: -1}, drop=True) - da.isel(
            {info[m].time_dim: 0}, drop=True
        )
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap_anom,
            ticks=ticks,
            cbar_extend="both",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig, ax, cs, label=info[m].units_label, ticks=ticks, extend="both"
    )

    suptitle = f"Change in {info[m].metric} {aep}% annual exceedance probability between {covariates[0]} and {covariates[1]}"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/aep_{aep:g}pct_trend_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_aep_empirical(info, var, aep=1):
    """Plot map of empirical AEP for a given threshold."""

    ari = eva.aep_to_ari(aep)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(models):
        da = eva.get_empirical_return_level(
            info[m].ds[var], ari, core_dim=info[m].time_dim
        )
        fig, ax[i + 1], cs = map_subplot(
            fig,
            ax[i + 1],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=info[m].cmap,
            ticks=info[m].ticks,
            cbar_extend=info[m].cbar_extend,
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=info[m].units_label,
        ticks=info[m].ticks,
        extend=info[m].cbar_extend,
    )

    suptitle = f"{info[m].metric} empirical {aep}% annual exceedance probability"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/aep_empirical_{aep:g}pct_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_new_record_probability(info, start_year, time_agg, n_years=10):
    """Plot map of the probability of breaking the obs record in the next X years."""
    # Get the event record (return period) for the obs data

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(info.keys()):
        # Get the event record (return period) for the obs data
        # N.B. The obs data in info[m] is already subset to the model time period
        record = info[m].obs_ds[info[m].var].reduce(func_dict[time_agg], dim="time")
        if info[m].is_model():
            record = general_utils.regrid(record, info[m].ds)
        cumulative_probability = nonstationary_new_record_probability(
            record, info[m].ds.dparams_ns, start_year, n_years, info[m].time_dim
        )
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            cumulative_probability * 100,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=plt.cm.BuPu,
            ticks=np.arange(0, 105, 5),
            cbar_extend="neither",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(fig, ax, cs, label="Probability [%]", extend="neither")

    suptitle = f"Probability of record breaking {info[m].metric} in the next {n_years} years ({start_year} to {start_year + n_years})"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/new_record_probability_{n_years}-year_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_new_record_probability_empirical(info, var, time_agg, n_years=10):
    """Plot map of the probability of breaking the obs record in the next X years."""
    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    for i, m in enumerate(models):
        _, cumulative_probability = new_record_probability_empirical(
            info[m].ds[var],
            info[m].obs_ds[var],
            n_years,
            time_agg,
            time_dim=info[m].time_dim,
            init_dim="init_date" if m != info[m].obs_name else "time",
        )

        fig, ax[i + 1], cs = map_subplot(
            fig,
            ax[i + 1],
            cumulative_probability * 100,
            title=f"{letters[i]} {info[m].title_name}",
            cbar=False,
            stippling=info[m].pval_mask,
            cmap=plt.cm.BuPu,
            ticks=np.arange(0, 105, 5),
            cbar_extend="neither",
        )

    # Hide empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(fig, ax, cs, label="Probability [%]", extend="neither")

    suptitle = f"Empirical probability of record breaking {info[m].metric} in the next {n_years} years"
    fig.suptitle(suptitle, fontsize=15)

    outfile = f"{info[m].fig_dir}/new_record_probability_{n_years}-year_empirical_{info[m].filestem}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_stability(
    info, var_dict, method="median", anomaly=False, ticks=None, **kwargs
):
    """Plot maps of lead-specific median/AEP for each model."""

    if method == "median":
        label = "Median"
        statistic = np.median

    elif method == "aep":
        label = "1% AEP"
        statistic = functools.partial(
            eva.empirical_return_level, return_period=eva.aep_to_ari(1)
        )

    if anomaly:
        label = f"{label} anomaly"
        extend = "both"
    else:
        extend = info[models[0]].cbar_extend

    fig, ax = plt.subplots(
        len(models), 9, figsize=(22, 22), subplot_kw=dict(projection=PlateCarree())
    )

    for i, m in enumerate(models):

        file = Path(var_dict[m]["metric_fcst"])
        da = fileio.open_dataset(str(file))[var_dict[m]["var"]]
        dx = da.stack(dict(sample=["ensemble", "init_date"]))
        dx = xr.apply_ufunc(
            statistic,
            dx,
            input_core_dims=[["sample"]],
            vectorize=True,
            dask="parallelized",
        )

        ci = info[m].ds[f"ci_{method}"]
        ci_mask = (dx < ci.isel(bounds=0, drop=True)) | (
            dx > ci.isel(bounds=1, drop=True)
        )

        if anomaly:
            dx_all = xr.apply_ufunc(
                statistic,
                da.stack(dict(sample=["ensemble", "init_date", "lead_time"])),
                input_core_dims=[["sample"]],
                vectorize=True,
                dask="parallelized",
            )
            dx = dx - dx_all

        # Shade background grey for lead times < min_lead
        min_lead = int(info[m].ds.min_lead_median.max().load().item())

        for j in dx.lead_time.values:
            fig, ax[i, j], cs = map_subplot(
                fig,
                ax[i, j],
                dx.isel(lead_time=j),
                title=f"{info[m].title_name} lead={j+1}",
                ticks=ticks,
                cbar=False,
                cmap=plt.cm.seismic,
                stippling=ci_mask.isel(lead_time=j),
                cbar_extend=extend,
            )

            if j < min_lead:
                # Set background color to grey
                ax[i, j] = plot_region_mask(
                    ax[i, j], regions_dict["not_australia"], facecolor="lightgrey"
                )

            if j != 0:
                ax[i, j].set_yticklabels([])
            if i != len(models) - 1:
                ax[i, j].set_xticklabels([])
        da.close()

    # Hide empty subplots
    for a in [a for a in ax.flatten() if not a.collections]:
        a.axis("off")

    suptitle = f"{label} of {info[m].metric} at each lead time"
    fig.suptitle(suptitle, fontsize=15)

    if anomaly:
        kwargs["hide_mid_ticklabels"] = True
    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"{label} [{info[m].units}]",
        extend=extend,
        ticks=ticks,
        **kwargs,
    )

    outfile = f"{info[m].fig_dir}/stability_{method}{'_anom' if anomaly else ''}_{info[m].metric}.png".lower()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    # # Define metric specific args
    # parser = argparse.ArgumentParser(description="Plot spatial maps of model data.")
    # parser.add_argument("--metric", type=str, help="Metric to plot (e.g., txx, rx1day)")
    # parser.add_argument(
    #     "--obs", type=str, help="Name of the observation dataset (e.g., AGCD, gridded_obs)"
    # )
    # parser.add_argument(
    #     "--bc",
    #     type=str,
    #     default=None,
    #     help="Bias correction method (e.g., None, additive, multiplicative)",
    # )
    # parser.add_argument(
    #     "--obs_config_file",
    #     type=str,
    #     help="Name of the observation dataset makefile (e.g., AGCD-CSIRO_r05_tasmax_config.mk)",
    # )
    # args = parser.parse_args()
    # metric, obs, bc, obs_config_file = (
    #     args.metric,
    #     args.obs,
    #     args.bc,
    #     args.obs_config_file,
    # )
    # # python3 spatial_maps.py --metric txx --obs AGCD --bc additive --obs_config_file AGCD-CSIRO_r05_tasmax_config.mk
    # # python3 spatial_maps.py --metric rx1day --obs AGCD --bc additive --obs_config_file AGCD-CSIRO_r05_precip_config.mk

    metric = "txx"
    obs_config_file = "AGCD-CSIRO_r05_tasmax_config.mk"
    obs = "AGCD"
    bc = [None, "additive", "multiplicative"][1]

    if metric == "txx":
        # Exclude HadGEM3-GC31-MM tasmax results
        models = np.array([m for m in models if m != "HadGEM3-GC31-MM"])

    # # Get variables from makefile (nested dictionary)
    var_dict = get_makefile_vars(metric, obs, obs_config_file=obs_config_file)

    # Filestem for figures and datatree
    filestem = f"{metric}_{var_dict[obs]['timescale']}_{var_dict[obs]['region']}"
    if bc is not None:
        filestem += f"_bias-corrected-{var_dict[obs]['obs_dataset']}-{bc}"
    dt_file = f"{var_dict[obs]['project_dir']}/data/datatree_{filestem}.nc"

    # Extract some variables from the dictionaries
    plot_dict = eval(var_dict[obs]["plot_dict"])
    var = var_dict[obs]["var"]
    time_agg = var_dict[obs]["time_agg"]
    fig_dir = Path(var_dict[obs]["fig_dir"]) / "paper"
    covariate_base = int(var_dict[obs]["covariate_base"])
    covariates = eval(var_dict[obs]["gev_trend_period"])

    # # Create/open datatree of all model and observation datasets
    if Path(dt_file).exists():
        dt = xr.open_datatree(dt_file)
    else:
        # Create a data tree using dict of {model: filenames["metric_fcst"]}
        # ~1.5 hour runtime
        data_dict = {}
        data_dict[f"obs/{obs}"] = open_obs_dataset(var_dict, obs)
        for i, model in enumerate(models):
            print(f"{i}. {model}")
            data_dict[f"model/{model}"] = open_model_dataset(var_dict[model], bc)
        dt = xr.DataTree.from_dict(data_dict)
        dt.to_netcdf(dt_file, compute=True)

    # Create nested dict of Infoset objects containing variables and datasets
    info = {}

    info[obs] = InfoSet(
        name=obs,
        file=var_dict[obs]["metric_obs"],
        obs_name=obs,
        ds=dt[f"obs/{obs}"].ds,
        obs_ds=dt[f"obs/{obs}"].ds,
        bias_correction=bc,
        fig_dir=fig_dir,
        pval_mask=None,
        filestem=filestem,  # Overrides filestem function
        **plot_dict,
    )

    for m in models:
        info[m] = InfoSet(
            name=m,
            obs_name=obs,
            file=var_dict[m]["metric_fcst"],
            ds=dt[f"model/{m}"].ds,
            obs_ds=subset_obs_dataset(dt[f"obs/{obs}"].ds, dt[f"model/{m}"].ds),
            pval_mask=dt[f"model/{m}"].ds.pval_mask,
            bias_correction=bc,
            fig_dir=fig_dir,
            filestem=filestem,  # Overrides filestem function
            **plot_dict,
        )


    # Plots
    # Stability (don't plot for diff bc)
    if metric == "txx":
        kwargs = dict(
            ticks=np.arange(-3.3, 3.5, 0.2),
            ticklabels=np.around(np.arange(-3.2, 3.4, 0.2), 1),
        )
    plot_stability(info, var_dict, method="median", anomaly=True, **kwargs)
    plot_stability(info, var_dict, method="aep", anomaly=True, **kwargs)
    plot_stability(
        info, var_dict, method="median", anomaly=False, ticks=info[obs].ticks
    )
    plot_stability(info, var_dict, method="aep", anomaly=False, ticks=info[obs].ticks)

    # Metric maximum/median
    plot_time_agg(info, var, "maximum")
    plot_time_agg(info, var, "median")
    plot_time_agg(info, var, "mean")
    plot_time_agg_subsampled(info, obs, "maximum", 1000)
    for anom in ["anom", "anom_pct", "anom_std", "anom_2000yr"]:
        plot_obs_anom(info, var, "maximum", metric=anom, covariate_base=covariate_base)
    ticks = np.arange(-5, 5.5, 0.5) if metric == "txx" and bc == "additive" else None
    plot_obs_anom(
        info, var, "median", metric="anom", covariate_base=covariate_base, ticks=ticks
    )

    # Seasonality/event year
    plot_event_month_mode(info)
    plot_event_year(info, var, "maximum", ticks=np.arange(1960, 2025, 5))

    # Independent lead time
    plot_min_independent_lead(info)

    # GEV/empirical
    for param in ["c", "loc0", "loc1", "scale0", "scale1"]:
        plot_nonstationary_gev_param(info, param)

    for aep in [1]:
        for covariate in covariates:
            plot_aep(info, covariate, aep=aep)
        plot_aep_trend(
            info,
            covariates,
            aep=aep,
            ticks=np.arange(-5, 5.5, 0.5) if metric == "txx" else None,
        )
        plot_aep_empirical(info, var, aep=aep)
    plot_new_record_probability(info, covariate_base, time_agg="maximum", n_years=10)
    plot_new_record_probability_empirical(info, var, time_agg="maximum", n_years=10)
