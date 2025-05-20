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
from cartopy.crs import PlateCarree
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
import functools
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from pathlib import Path
from scipy.stats import mode, median_abs_deviation
import string
import xarray as xr
import xesmf as xe

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
plt.rcParams["figure.dpi"] = 300
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


# -------------------------------------------------------------------------
# Generic plotting functions
# -------------------------------------------------------------------------


def map_subplot(
    fig,
    ax,
    data,
    region="aus_states_territories",
    hatching=None,
    title=None,
    ticks=None,
    ticklabels=None,
    cmap=plt.cm.viridis,
    norm=None,
    plot_cbar=False,
    cbar_label=None,
    extend="neither",
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
    if ticks is not None:
        if isinstance(ticks, (list, np.ndarray)):
            # Stolen from acs_plotting_maps
            if ticklabels is None or (len(ticklabels) == len(ticks) - 1):
                norm = BoundaryNorm(ticks, cmap.N + 1, extend=extend)
            else:
                print("Not implemented yet")

    cs = ax.pcolormesh(data.lon, data.lat, data, cmap=cmap, norm=norm)

    if hatching is not None:
        ax = add_hatching(ax, hatching)

    if mask_not_australia:
        ax = plot_region_mask(ax, regions_dict["not_australia"])

    if plot_cbar:
        fig.colorbar(cs, extend=extend, label=cbar_label, ticks=ticks, **cbar_kwargs)

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


def add_hatching(ax, hatching, **kwargs):
    """Add hatching to contour plot where hatching==True."""
    ax.contourf(
        hatching.lon,
        hatching.lat,
        hatching,
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
    tick_interval=1,
    **kwargs,
):
    """Add a shared colorbar to a figure with multiple subplots."""
    norm = cs.norm
    if ticks is None and hasattr(norm, "boundaries"):
        ticks = norm.boundaries
    if "extend" not in kwargs:
        kwargs["extend"] = norm.extend

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
        if ticklabels is not None and not isinstance(ticklabels, (list, np.ndarray)):
            ticklabels = np.array(ticks[:-1] + (np.diff(ticks)[0] / 2))
            if (ticklabels.astype(int) == ticklabels).all():
                ticklabels = ticklabels.astype(int)

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

    # Hide every other tick label
    axis = "y" if orientation == "vertical" else "x"
    labels = eval(f"cbar.ax.{axis}axis.get_ticklabels()")
    for i, label in enumerate(labels):
        # Only show labels at specific interval
        if i % tick_interval != 0:
            label.set_visible(False)
    return cbar


def plot_region_border(ax, region, **kwargs):
    """Plot shapefile region borders."""

    ec = kwargs.pop("ec", "k")  # Default edge color
    lw = kwargs.pop("lw", 0.5)  # Default line width

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


def plot_region_mask(ax, region, **kwargs):
    """Mask data outside of the region."""
    facecolor = kwargs.pop("facecolor", "white")
    ax.add_geometries(
        region,
        crs=PlateCarree(),
        linewidth=0,
        facecolor=facecolor,
        **kwargs,
    )
    return ax


def extra_subplot_formatting(ax):
    # Increase border width of observations & multi-model subplot
    [a.set_linewidth(2) for a in ax[0].spines.values()]
    [a.set_edgecolor("midnightblue") for a in ax[0].spines.values()]
    [a.set_linewidth(2) for a in ax[1].spines.values()]

    # Hide any empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")
    return ax


def add_aus_state_labels(ax, **kwargs):
    """Add Australian states and territories labels to a map."""
    regions = regions_dict["aus_states_territories"]
    regions = regions[:-1]  # exclude "other territories"

    for name, centroid in zip(regions.ABBREV, regions.centroid):
        x, y = centroid.x, centroid.y
        if name not in ["ACT", "VIC", "TAS"]:
            # Add the text label at the centroid location
            ax.text(x, y - 0.4, name, ha="center", va="center", **kwargs)
        else:
            # Annotate name to the right with line pointing to centroid
            if name == "ACT":
                dx, dy = 4.8, -0.5
            elif name == "VIC":
                dx, dy = 8.5, -2.6
            elif name == "TAS":
                dx, dy = 6, -0.7
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(x + dx, y + dy),
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", lw=1, shrinkA=0, shrinkB=-2),
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
    stability_n_resamples=10000,
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

    # Similarity test (for hatching)
    similarity_ds = fileio.open_dataset(similarity_file)
    ds_independence = xr.open_dataset(kws["independence_file"], decode_times=time_coder)
    # Non-stationary GEV parameters
    dparams_ns = xr.open_dataset(gev_params_nonstationary_file)[var]

    # Calculate stability confidence intervals (for median and 1% AEP)
    if bc is not None:
        da = fileio.open_dataset(kws[f"metric_fcst"])[var]
    else:
        da = model_ds[var]
    ci_median = get_stability_ci(
        da, "median", confidence_level=0.99, n_resamples=stability_n_resamples
    )
    ci_aep = get_stability_ci(
        da, "aep", confidence_level=0.99, n_resamples=stability_n_resamples, aep=1
    )

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


def shared_grid_regridder(ds, method="conservative"):
    """Add dataset-specific regridder to each InfoSet instance in info."""
    # Create a 1x1 degree grid over Australia
    grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-44, -11, 1.0), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(112, 155, 1.0), {"units": "degrees_east"}),
        }
    )
    regridder = xe.Regridder(ds, grid, method)

    return regridder


def multimodel_avg(info, da_list, func=np.median, **kwargs):
    """Regrid arrays to common grid and return the multi-model average."""

    dr_list = []
    for i, m in enumerate(models):
        # Regrid each DataArray to the common grid using the model's regridder
        dr_list.append(info[m].regridder(da_list[i]))

    dr = xr.concat(dr_list, dim="model")  # Concat along axis=0
    if func is not None:
        dr = dr.reduce(func, "model", **kwargs)
    return dr


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


def plot_time_agg(info, var, time_agg, plot_dict):
    """Plot time-aggregated data for each model and observation dataset."""

    cbar_kwargs = dict(
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):

        da = info[m].ds[var].reduce(func_dict[time_agg], dim=info[m].time_dim)
        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig, ax[i], dm, title=f"{letters[i]} Multi-model median", **cbar_kwargs
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=plot_dict["units_label"],
        **cbar_kwargs,
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/{time_agg}_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_obs_anom(
    info,
    obs,
    var,
    time_agg,
    metric,
    covariate_base,
    plot_dict,
):
    """Plot map of soft-record metric (e.g., anomaly) between model and observation."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    # Plot obs time agg (not anomaly)
    i = 0
    da_obs = info[obs].ds[var].reduce(func_dict[time_agg], dim=info[obs].time_dim)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        da_obs,
        title=f"{letters[i]} Observed {time_agg} {plot_dict['metric']}",
        hatching=info[obs].pval_mask,
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    # Create an inset axes for the colorbar [x0, y0, width, height]
    cax = ax[i].inset_axes([0.04, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
    cbar.ax.set_title(f"{plot_dict['metric']} [{info[obs].units}]")

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(models):
        i += 2  # Leave ax[0-1] for obs & multi-model median

        da, kwargs = soft_record_metric(
            info[m].ds[var],
            info[m].obs_ds[var],
            time_agg,
            metric,
            plot_dict,
            time_dim=info[m].time_dim,
            dparams_ns=info[m].ds["dparams_ns"],
            covariate_base=covariate_base,
        )
        da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=kwargs["cmap"],
            ticks=kwargs["ticks"],
            extend=kwargs["cbar_extend"],
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=kwargs["cmap"],
        ticks=kwargs["ticks"],
        extend=kwargs["cbar_extend"],
    )

    kwargs["cbar_label"] = kwargs["cbar_label"].replace("\n", " ")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=kwargs["cbar_label"],
        ticks=kwargs["ticks"],
        extend=kwargs["cbar_extend"],
        tick_interval=kwargs["tick_interval"],
        ticklabels=True,
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = kwargs["title"].replace("\n", " ")
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/{time_agg}_{metric}_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_time_agg_subsampled(
    info,
    obs,
    time_agg,
    plot_dict,
    resamples=1000,
):
    """Plot map of observation-sized subsample of data (sample median of time-aggregate).
    Also plot the anomaly of the subsampled data minus regrided obs data.
    """

    n_obs_samples = info[obs].obs_ds[info[obs].var].time.size
    print(f"Number of obs samples: {n_obs_samples}")

    cbar_kwargs = dict(
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    # Plot obs time agg (not subsampled)
    i = 0
    da_obs = info[obs].ds[var].reduce(func_dict[time_agg], dim=info[obs].time_dim)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        da_obs,
        title=f"{letters[i]} Observed {time_agg} {plot_dict['metric']}",
        **cbar_kwargs,
    )

    da_list = []  # Store data arrays for the multi-model median & anomaly plot
    for i, m in enumerate(models):
        i += 2  # Leave ax[0-1] for obs and the multi-model median
        da = resample_subsample(info[m], info[m].ds, time_agg, n_obs_samples, resamples)
        da_list.append(da)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig, ax[i], dm, title=f"{letters[i]} Multi-model median", **cbar_kwargs
    )

    add_shared_colorbar(fig, ax, cs, label=plot_dict["units_label"], **cbar_kwargs)

    ax = extra_subplot_formatting(ax)

    # suptitle = f"{plot_dict['metric']} {time_agg} in obs-sized subsample (median of {resamples} resamples)"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = (
        f"{plot_dict['fig_dir']}/{time_agg}_subsampled_{plot_dict['filestem']}.png"
    )
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()

    # Plot anomaly of subsampled data minus regrided obs data
    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    cbar_kwargs = dict(
        cmap=plot_dict["cmap_anom"],
        ticks=plot_dict["ticks_anom"],
        extend="both",
    )

    # Plot obs time agg (not subsampled)
    i = 0
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        da_obs,
        title=f"{letters[i]} Observed {time_agg} {plot_dict['metric']}",
        hatching=info[obs].pval_mask,
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    # Create an inset axes for the colorbar [x0, y0, width, height]
    cax = ax[i].inset_axes([0.04, 0.1, 0.7, 0.05])
    cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
    cbar.ax.set_title(f"{plot_dict['metric']} [{info[obs].units}]")

    da_anom_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(models):
        da = da_list[i]
        i += 2  # Leave ax[0-1] for obs and the multi-model median
        da_obs_regrid = general_utils.regrid(da_obs, da)
        da = da - da_obs_regrid
        da_anom_list.append(da)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_anom_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        **cbar_kwargs,
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=plot_dict["units_label"],
        ticklabels=True,
        **cbar_kwargs,
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = f"{plot_dict['metric']} {time_agg} in obs-sized subsample (median of {resamples} resamples; observed anomaly)"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = (
        f"{plot_dict['fig_dir']}/{time_agg}_subsampled_anom_{plot_dict['filestem']}.png"
    )
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_event_month_mode(info, plot_dict, min_count=4, add_labels=True):
    """Plot map of the most common month when event occurs."""

    cbar_kwargs = dict(
        cmap=month_cmap,
        ticks=np.arange(0.5, 13.5),
        ticklabels=list(calendar.month_name)[1:],
        extend="neither",
    )

    # Add white space for ACT, TAS labels
    map_kwargs = dict(xlim=(112.5, 157)) if add_labels else {}

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model mode
    for i, m in enumerate(info.keys()):

        da = xr.DataArray(
            mode(info[m].ds.event_time.dt.month, axis=0).mode,
            coords=dict(lat=info[m].ds.lat, lon=info[m].ds.lon),
            dims=["lat", "lon"],
        )
        if m in models:
            i += 1  # Leave ax[1] for the multi-model mode
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            **cbar_kwargs,
            **map_kwargs,
        )

    # Multi-model mode
    dr_list = []
    for i, m in enumerate(models):
        regridder = shared_grid_regridder(info[m].ds, method="nearest_s2d")
        dr_list.append(regridder(da_list[i]))

    dm = xr.concat(dr_list, dim="model")
    mm_mode = mode(dm, axis=0, nan_policy="omit")
    dm = xr.DataArray(
        mm_mode.mode, coords=dict(lat=dm.lat, lon=dm.lon), dims=("lat", "lon")
    )
    counts = xr.DataArray(mm_mode.count, coords=dm.coords, dims=dm.dims)

    i = 1
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model mode",
        hatching=counts < min_count,
        **cbar_kwargs,
        **map_kwargs,
    )
    if add_labels:
        # Add Australian state and territory labels to the first map
        add_aus_state_labels(ax[0], color="black", fontsize=12, fontweight="bold")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"Most common month of {plot_dict['metric']} occurrence",
        **cbar_kwargs,
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/month_mode_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_event_month_max(info, plot_dict, min_count=4):
    """Plot map of the most common month when event occurs."""
    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model mode
    for i, m in enumerate(info.keys()):
        # Change the time dimension to event year
        dt = info[m].ds[var].copy().compute()
        dt.coords[info[m].time_dim] = dt.event_time.dt.month

        # Get the year of the maximum or minimum event
        if time_agg == "maximum":
            da = dt.idxmax(info[m].time_dim)
        if m in models:
            i += 1  # Leave ax[1] for the multi-model mode
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            cmap=month_cmap,
            ticks=np.arange(0.5, 13.5),
            ticklabels=list(calendar.month_name)[1:],
            extend="neither",
        )

    # Multi-model mode
    dr_list = []
    for i, m in enumerate(models):
        regridder = shared_grid_regridder(info[m].ds, method="nearest_s2d")
        dr_list.append(regridder(da_list[i]))

    dm = xr.concat(dr_list, dim="model")
    mm_mode = mode(dm, axis=0, nan_policy="omit")
    dm = xr.DataArray(
        mm_mode.mode, coords=dict(lat=dm.lat, lon=dm.lon), dims=("lat", "lon")
    )
    counts = xr.DataArray(mm_mode.count, coords=dm.coords, dims=dm.dims)

    i = 1
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model mode",
        hatching=counts < min_count,
        cmap=month_cmap,
        ticks=np.arange(0.5, 13.5),
        ticklabels=list(calendar.month_name)[1:],
        extend="neither",
    )
    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"Month of maximum {plot_dict['metric']} event",
        extend="neither",
        ticks=np.arange(0.5, 13.5),
        ticklabels=list(calendar.month_abbr)[1:],
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/month_max_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_event_year(
    info, var, time_agg, plot_dict, ticks=np.arange(1960, 2025, 5), min_count=4
):
    """Plot map of the year of the maximum or minimum event."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    da_list = []  # Store data arrays for the multi-model mode

    for i, m in enumerate(info.keys()):

        # Change the time dimension to event year
        dt = info[m].ds[var].copy().compute()
        dt.coords[info[m].time_dim] = dt.event_time.dt.year

        # Get the year of the maximum or minimum event
        if time_agg == "maximum":
            da = dt.idxmax(info[m].time_dim)
        elif time_agg == "minimum":
            da = dt.idxmin(info[m].time_dim)

        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            # hatching=info[m].pval_mask,
            cmap=cmap_dict["inferno"],
            ticks=ticks,
            extend="max",
        )

    # Multi-model mode (position after obs & before models)
    # add hatching where mode count is less than min_count
    dr_list = []
    tick_bins = np.concatenate([ticks, [np.inf]])  # Add an extra bin to the right
    for i, m in enumerate(models):
        bin_indices = np.digitize(da_list[i], tick_bins)
        da = tick_bins[bin_indices - 1]
        da = xr.DataArray(da, da_list[i].coords, dims=da_list[i].dims)

        regridder = shared_grid_regridder(info[m].ds, method="nearest_s2d")
        dr_list.append(regridder(da))

    dm = xr.concat(dr_list, dim="model")
    # Get the multi-model mode and count for each grid cell
    mmm = mode(dm, axis=0, nan_policy="omit")
    dm = xr.DataArray(
        mmm.mode, coords=dict(lat=dm.lat, lon=dm.lon), dims=("lat", "lon")
    )
    # Counts for each mode
    counts = xr.DataArray(mmm.count, coords=dm.coords, dims=dm.dims)

    i = 1
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model mode",
        cmap=cmap_dict["inferno"],
        ticks=ticks,
        extend="max",
        hatching=counts < min_count,
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"Year of {time_agg} {plot_dict['metric']}",
        extend="max",
        ticks=ticks,
        ticklabels=None,
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/year_{time_agg}_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_aep(info, plot_dict, covariate, aep=1):
    """Plot maps of AEP for a given threshold (at a covariate value)."""

    ari = eva.aep_to_ari(aep)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):
        # _covariate = xr.DataArray([covariate], dims=info[m].time_dim)
        da = eva.get_return_level(ari, info[m].ds.dparams_ns, covariate)
        if m in models:
            i += 1
            da_list.append(da)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=plot_dict["cmap"],
            ticks=plot_dict["ticks"],
            extend=plot_dict["cbar_extend"],
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=plot_dict["units_label"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
        ticklabels=True,
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = f"{plot_dict['metric']} {aep}% annual exceedance probability for {covariate} [{plot_dict['units']}]"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/aep_{aep:g}pct_{covariate:.0f}_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")


def plot_aep_trend(info, plot_dict, covariates, aep=1):
    """Plot map of the change in AEP between two covariate (list) values."""

    ari = eva.aep_to_ari(aep)

    cbar_kwargs = dict(
        cmap=plot_dict["cmap_anom"], ticks=plot_dict["ticks_trend"], extend="both"
    )

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []
    for i, m in enumerate(info.keys()):
        _covariates = xr.DataArray(covariates, dims=info[m].time_dim)
        da = eva.get_return_level(ari, info[m].ds.dparams_ns, _covariates)
        da = da.isel({info[m].time_dim: -1}, drop=True) - da.isel(
            {info[m].time_dim: 0}, drop=True
        )
        if m in models:
            i += 1
            da_list.append(da)
        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig, ax[i], dm, title=f"{letters[i]} Multi-model median", **cbar_kwargs
    )

    add_shared_colorbar(
        fig, ax, cs, label=plot_dict["units_label"], ticklabels=True, **cbar_kwargs
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = f"Change in {plot_dict['metric']} {aep}% annual exceedance probability between {covariates[0]} and {covariates[1]}"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/aep_{aep:g}pct_trend_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_aep_empirical(info, plot_dict, var, aep=1):
    """Plot map of empirical AEP for a given threshold."""

    ari = eva.aep_to_ari(aep)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):
        da = eva.get_empirical_return_level(
            info[m].ds[var], ari, core_dim=info[m].time_dim
        )
        if m in models:
            i += 1
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=plot_dict["cmap"],
            ticks=plot_dict["ticks"],
            extend=plot_dict["cbar_extend"],
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=plot_dict["cmap"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=plot_dict["units_label"],
        ticks=plot_dict["ticks"],
        extend=plot_dict["cbar_extend"],
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = f"{plot_dict['metric']} empirical {aep}% annual exceedance probability"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = (
        f"{plot_dict['fig_dir']}/aep_empirical_{aep:g}pct_{plot_dict['filestem']}.png"
    )
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_new_record_probability(info, plot_dict, start_year, time_agg, n_years=10):
    """Plot map of the probability of breaking the obs record in the next X years."""
    # Get the event record (return period) for the obs data

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()
    da_list = []  # Store data arrays for the multi-model median

    for i, m in enumerate(info.keys()):
        # Get the event record (return period) for the obs data
        # N.B. The obs data in info[m] is already subset to the model time period
        record = info[m].obs_ds[info[m].var].reduce(func_dict[time_agg], dim="time")
        if info[m].is_model():
            record = general_utils.regrid(record, info[m].ds)
        cumulative_probability = nonstationary_new_record_probability(
            record, info[m].ds.dparams_ns, start_year, n_years, info[m].time_dim
        )
        da = cumulative_probability * 100
        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)  # Only append model data

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=plt.cm.BuPu,
            ticks=np.arange(0, 105, 5),
            extend="neither",
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=plt.cm.BuPu,
        ticks=np.arange(0, 105, 5),
        extend="neither",
    )

    add_shared_colorbar(fig, ax, cs, label="Probability [%]", extend="neither")

    ax = extra_subplot_formatting(ax)

    # suptitle = f"Probability of record breaking {plot_dict['metric']} in the next {n_years} years ({start_year} to {start_year + n_years})"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/new_record_probability_{n_years}-year_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_new_record_probability_empirical(info, plot_dict, var, time_agg, n_years=10):
    """Plot map of the probability of breaking the obs record in the next X years."""

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):
        _, cumulative_probability = new_record_probability_empirical(
            info[m].ds[var],
            info[m].obs_ds[var],
            n_years,
            time_agg,
            time_dim=info[m].time_dim,
            init_dim="init_date" if m != info[m].obs_name else "time",
        )
        da = cumulative_probability * 100

        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)  # Only append model data

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=plt.cm.BuPu,
            ticks=np.arange(0, 105, 5),
            extend="neither",
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=plt.cm.BuPu,
        ticks=np.arange(0, 105, 5),
        extend="neither",
    )

    add_shared_colorbar(fig, ax, cs, label="Probability [%]", extend="neither")

    ax = extra_subplot_formatting(ax)

    # suptitle = f"Empirical probability of record breaking {plot_dict['metric']} in the next {n_years} years"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/new_record_probability_{n_years}-year_empirical_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_obs_ari(
    info,
    plot_dict,
    var,
    obs,
    covariate_base,
    time_agg="maximum",
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    obs_ds : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, optional
        Model dataset
    dparams_ns : xarray.DataArray
        Non-stationary GEV parameters
    covariate_base : int
        Covariate for non-stationary GEV parameters (e.g., single year)
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """
    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    cbar_kwargs = dict(
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=10000),
        extend="max",
    )
    obs_da = info[obs].ds[var].reduce(func_dict[time_agg], dim="time")

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):
        if m != obs:
            obs_da_agg = general_utils.regrid(obs_da, info[m].ds[var])
        else:
            obs_da_agg = obs_da

        da = xr.apply_ufunc(
            eva.get_return_period,
            obs_da_agg,
            info[m].ds.dparams_ns,
            input_core_dims=[[], ["dparams"]],
            output_core_dims=[[]],
            kwargs=dict(
                covariate=xr.DataArray([covariate_base], dims=info[m].time_dim)
            ),
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float64"],
        )

        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)  # Only append model data

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Multi-model median
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig, ax[i], dm, title=f"{letters[i]} Multi-model median", **cbar_kwargs
    )

    add_shared_colorbar(
        fig, ax, cs, label=f"Average recurrence interval [years]", **cbar_kwargs
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/obs_ari_{time_agg}_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_metric_variability(info, var, plot_dict):
    """Plot the median absolution deviation of the metric."""

    cbar_kwargs = dict(
        cmap=plt.cm.viridis,
        ticks=np.arange(0, 4.5, 0.5),
        extend="max",
    )
    statistic = median_abs_deviation
    kwargs = dict(center=np.median)

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):

        da = info[m].ds[var].reduce(statistic, dim=info[m].time_dim, **kwargs)
        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            **cbar_kwargs,
        )

    # Show obs time agg (not subsampled)
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig, ax[i], dm, title=f"{letters[i]} Multi-model median", **cbar_kwargs
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"Median absolute deviation [{plot_dict['units']}]",
        **cbar_kwargs,
    )

    ax = extra_subplot_formatting(ax)

    outfile = f"{plot_dict['fig_dir']}/median_abs_deviation_{plot_dict['filestem']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_nonstationary_gev_param(info, param, plot_dict):
    """Plot map of GEV location and scale parameter trends.

    Parameters
    ----------
    info : dict
        Dictionary of InfoSet instances for each model and observation dataset.
    param : {'c', 'loc0', 'loc1', 'scale0', 'scale1'}

    Notes
    -----
    * The trend parameters are multiplied by 10 to convert to per decade.
    * The location and scale intercept parameter ticks are defined here, but
    the ticks for the trend parameters are defined in the metric config file.

    """

    # Nested dict of parameter plotting variables
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
            ticks=plot_dict["ticks_param_trend"]["location"],
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
            ticks=plot_dict["ticks_param_trend"]["scale"],
            cmap=plt.cm.RdBu_r,
        ),
    }

    assert param in param_dict_dict.keys(), f"Unknown parameter: {param}"
    # Get parameter-specific dictionary
    param_dict = param_dict_dict[param]

    fig, ax = plt.subplots(3, 4, subplot_kw=dict(projection=PlateCarree()))
    ax = ax.flatten()

    da_list = []  # Store data arrays for the multi-model median
    for i, m in enumerate(info.keys()):

        da = info[m].ds.dparams_ns.sel(dparams=param)

        if param in ["loc1", "scale1"]:
            da = da * 10  # Convert to per decade

        if m in models:
            i += 1  # Leave ax[1] for the multi-model median
            da_list.append(da)  # Only append model data

        fig, ax[i], cs = map_subplot(
            fig,
            ax[i],
            da,
            title=f"{letters[i]} {info[m].title_name}",
            hatching=info[m].pval_mask,
            cmap=param_dict["cmap"],
            ticks=param_dict["ticks"],
            extend=param_dict["extend"] if "extend" in param_dict else "both",
        )

    # Multi-model median (position after obs & before models)
    i = 1
    dm = multimodel_avg(info, da_list)
    fig, ax[i], cs = map_subplot(
        fig,
        ax[i],
        dm,
        title=f"{letters[i]} Multi-model median",
        cmap=param_dict["cmap"],
        ticks=param_dict["ticks"],
        extend=param_dict["extend"] if "extend" in param_dict else "both",
    )

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"{param_dict['name'].capitalize()} parameter",
        ticks=param_dict["ticks"],
    )

    ax = extra_subplot_formatting(ax)

    # suptitle = f"{plot_dict['metric']} GEV distribution {param_dict['name']} parameter"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = (
        f"{plot_dict['fig_dir']}/gev_{param_dict['name']}_{plot_dict['filestem']}.png"
    )
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_min_independent_lead(info, plot_dict):
    """Plot map of the minimum independent lead time for each model.

    The minimum independent lead time is the first lead time in which the
    ensemble mean correlation coefficient is within the 99% confidence interval.

    Spatial map of the minimum independent lead time for each model and init
    month with the spatial median lead in lower left corner of each subplot.
    Lead time 1 is the first lead time instead of zero.
    Plot doesn't include obs (or a blank space) or the multi-model median.
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
            fig, ax[i], cs = map_subplot(
                fig,
                ax[i],
                dx,
                title=f"{letters[i]} {info[m].name} ({calendar.month_name[month]} starts)",
                cmap=plt.cm.viridis,
                ticks=ticks,
                ticklabels=ticklabels,
                extend="neither",
            )

            # Add box with value of min lead spatial median in lower left corner
            min_lead_median = da.min_lead_median.load()
            if "month" in min_lead_median.dims:
                min_lead_median = min_lead_median.sel(month=month)

            ax[i].text(
                0.05,
                0.05,
                f"median={min_lead_median.item() + 1:.0f}",
                ha="left",
                va="bottom",
                transform=ax[i].transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="black",
                    facecolor="white",
                    alpha=0.8,
                ),
            )
            i += 1

    # Hide any empty subplots
    for a in [a for a in ax if not a.collections]:
        a.axis("off")

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=f"First independent {plot_dict['metric']} lead time",
        extend="neither",
        ticks=ticks,
        ticklabels=ticklabels,
    )

    outfile = f"{plot_dict['fig_dir']}/independence_{plot_dict['filestem_no_bc']}.png"
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_stability(
    info, var_dict, plot_dict, method="aep", anomaly=False, ticks=None, **kwargs
):
    """Plot maps of lead-specific median/AEP for each model."""

    if method == "median":
        label = f"Median {plot_dict['metric']}"
        statistic = np.median

    elif method == "aep":
        label = f"{plot_dict['metric']} 1% Annual Exceedence Probability"
        statistic = functools.partial(
            eva.empirical_return_level, return_period=eva.aep_to_ari(1)
        )

    if anomaly:
        label = f"{label} anomaly"
        extend = "both"
        kwargs["tick_interval"] = 2
        if ticks is None:
            ticks = plot_dict["ticks_anom"]
    else:
        extend = plot_dict["cbar_extend"]
        if ticks is None:
            ticks = plot_dict["ticks"]

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
                hatching=ci_mask.isel(lead_time=j),
                ticks=ticks,
                cmap=plt.cm.seismic if anomaly else plot_dict["cmap"],
                extend=extend,
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

    add_shared_colorbar(
        fig,
        ax,
        cs,
        label=plot_dict["units_label"],
        extend=extend,
        ticks=ticks,
        **kwargs,
    )

    # Hide any empty subplots
    for a in [a for a in ax.flatten() if not a.collections]:
        a.axis("off")

    # suptitle = f"{label} of {plot_dict['metric']} at each lead time"
    # fig.suptitle(suptitle, fontsize=15)

    outfile = f"{plot_dict['fig_dir']}/stability_{method}{'_anom' if anomaly else ''}_{plot_dict['filestem_no_bc']}.png".lower()
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
    plt.close()


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
    covariate_base = int(var_dict[obs]["covariate_base"])
    covariates = eval(var_dict[obs]["gev_trend_period"])
    plot_dict["fig_dir"] = Path(var_dict[obs]["fig_dir"]) / "multimodel"
    plot_dict["filestem"] = filestem
    plot_dict["filestem_no_bc"] = filestem.split("_bias")[0]
    plot_dict_avg = plot_dict.copy()
    stability_kwargs = {}
    stability_anom_kwargs = {}

    if metric == "txx":
        plot_dict["cmap"] = cmap_dict["ipcc_temp_seq"]
        if bc is None:
            plot_dict_avg["ticks"] = np.arange(22, 54 + 2, 2)
            plot_dict["ticks"] = np.arange(24, 66 + 2, 2)
        else:
            plot_dict_avg["ticks"] = np.arange(22, 54 + 2, 2)
            plot_dict["ticks"] = np.arange(32, 58 + 2, 2)

        stability_anom_kwargs = dict(
            ticks=np.arange(-3.3, 3.5, 0.2),
            ticklabels=np.around(np.arange(-3.2, 3.4, 0.2), 1),
        )

    # # Create/open datatree of all model and observation datasets
    if Path(dt_file).exists():
        dt = xr.open_datatree(dt_file)
    else:
        # Create a data tree using dict of {model: filenames["metric_fcst"]}
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
        pval_mask=None,
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
            **plot_dict,
        )
        info[m].regridder = shared_grid_regridder(info[m].ds, method="conservative")

    # Plots
    # Metric maximum/median
    plot_time_agg(info, var, "maximum", plot_dict)
    plot_time_agg(info, var, "median", plot_dict_avg)
    plot_time_agg_subsampled(info, obs, "maximum", plot_dict, 10000)
    for anom in ["anom", "anom_pct", "anom_std", "anom_2000yr"]:
        plot_obs_anom(info, obs, var, "maximum", anom, covariate_base, plot_dict)
    plot_obs_anom(
        info,
        obs,
        var,
        "median",
        "anom",
        covariate_base,
        plot_dict_avg,
    )

    # Seasonality/event year
    plot_event_month_mode(info, plot_dict, add_labels=True)
    plot_event_month_max(info, plot_dict)
    plot_event_year(info, var, "maximum", plot_dict, ticks=np.arange(1960, 2025, 5))
    plot_metric_variability(info, var, plot_dict)

    aep = 1
    plot_aep(info, plot_dict, covariates[-1], aep=aep)
    plot_aep_trend(
        info,
        plot_dict,
        covariates,
        aep=aep,
    )
    plot_aep_empirical(info, plot_dict, var, aep=aep)

    plot_new_record_probability(
        info, plot_dict, covariate_base, time_agg="maximum", n_years=10
    )
    plot_new_record_probability_empirical(
        info, plot_dict, var, time_agg="maximum", n_years=10
    )
    plot_obs_ari(
        info,
        plot_dict,
        var,
        obs,
        covariate_base,
        time_agg="maximum",
    )
    # GEV/empirical
    for param in ["c", "loc0", "loc1", "scale0", "scale1"]:
        plot_nonstationary_gev_param(info, param, plot_dict)

    if bc is None:
        # Independent lead time
        plot_min_independent_lead(info, plot_dict)

        # Stability (don't plot for diff bc)
        for method in ["aep", "median"]:
            plot_stability(info, var_dict, plot_dict, method, anomaly=False)
            plot_stability(
                info,
                var_dict,
                plot_dict,
                method=method,
                anomaly=True,
                **stability_anom_kwargs,
            )
