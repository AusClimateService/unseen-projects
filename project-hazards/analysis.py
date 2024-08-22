# -*- coding: utf-8 -*-
"""Functions for plotting Australian hazard data using the acs_plotting_maps module."""

import calendar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from dataclasses import dataclass
from datetime import datetime
import geopandas as gp
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MaxNLocator, FixedLocator
from collections import namedtuple
import numpy as np
from pathlib import Path
from scipy.stats import genextreme, mode
import sys
import xarray as xr
import xesmf as xe
import yaml

# from unseen.fileio import open_dataset, open_mfforecast
from unseen.spatial_selection import select_shapefile_regions
from unseen import independence, stability, eva, general_utils

# sys.path.append("/g/data/xv83/as3189/plotting_maps/")
from acs_plotting_maps import plot_acs_hazard, regions_dict, cmap_dict, tick_dict

# # Load the configuration file
# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)

# # Access the paths from the configuration
# home = Path(config["paths"]["output"])

home = Path("/g/data/xv83/unseen-projects/outputs/hazards")
dcpp_models = np.array(
    [
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

func_dict = {
    "mean": np.mean,
    "median": np.median,
    "maximum": np.nanmax,
    "minimum": np.nanmin,
    "sum": np.sum,
}

Hazard = namedtuple(
    "Hazard",
    "index name var var_name units units_label timescale obs cmap ticks ticks_anom ticks_param_trend",
)

hazard_dict = {
    "txx": Hazard(
        "txx",
        "TXx",
        "tasmax",
        "Temperature",
        units="°C",
        units_label="Temperature [°C]",
        timescale="annual-jul-to-jun",
        obs="AGCD",
        cmap=cmap_dict["tasmax"],
        # cmap_anom=cmap_dict["anom"],
        ticks=np.arange(20, 60 + 5, 5),
        ticks_anom=tick_dict["tas_anom_mon"],
        ticks_param_trend={
            "location": np.arange(-0.08, 0.09, 0.02),
            "scale": np.arange(-0.015, 0.016, 0.005),
        },
    )
}


plot_kwargs = dict(
    name="ncra_regions",
    # regions=regions_dict["aus_states_territories"],
    mask_not_australia=True,
    figsize=[8 * 0.7, 6 * 0.7],
    area_linewidth=0.1,
    xlim=(114, 162),
    ylim=(-43, -8),
    label_states=False,
    contourf=False,
    contour=False,
    select_area=None,
    land_shadow=True,
    watermark=None,
)


def preprocess_dataset(ds):
    # Apply Australian land-sea mask
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    ds = select_shapefile_regions(ds, gdf, overlap_fraction=0.5)

    # Format event time as datetime
    if "event_time" in ds:
        ds["event_time"] = ds.event_time.astype(dtype="datetime64[ns]")
    return ds


def get_dcpp_model_dataset(dataset, hazard):
    file = list(home.rglob(f"data/{hazard.index}_{dataset}*_{hazard.timescale}_aus.nc"))
    ds = xr.open_dataset(str(file[0]), use_cftime=True)
    ds = ds.isel(lead_time=slice(1, None), drop=True)  # drop first lead time
    ds = preprocess_dataset(ds)
    return ds


def get_obs_dataset(hazard):
    # Observational data
    file = (
        f"{home}/data/{hazard.index}_AGCD-CSIRO_r05_1910-2023_{hazard.timescale}_aus.nc"
    )
    ds_obs = xr.open_dataset(file, use_cftime=True, chunks="auto")
    # Drop years before model data is available
    ds_obs = ds_obs.sel(time=slice("1960", None))
    ds_obs = preprocess_dataset(ds_obs)
    return ds_obs


def date_range_str(time):
    """Return date range 'DD month YYYY' string from time coordinate."""
    # @todo: test for YE-DEC
    # First and last year
    year = [f(time.dt.year.values) for f in [np.min, np.max]]
    # Index of year end month (assumes time agg index by YEAR_END_MONTH)
    YE_ind = [time.dt.month[0].item() + i for i in [1, 0]]
    # First and last month name
    mon = [list(calendar.month_name)[i] for i in YE_ind]

    day = [1, calendar.monthrange(year[1], YE_ind[1])[-1]]
    date_range = " to ".join([f"{day[i]} {mon[i]} {year[i]}" for i in [0, 1]])
    return date_range


def regrid_like(ds, ds_like, method="conservative"):
    """Regrid `ds` to the grid of `ds_like` using xESMF."""
    assert ds.dims[-2:] == ("lat", "lon"), "Last two dimensions must be lat and lon"
    grid_in = xr.Dataset(coords=dict(lat=ds.lat, lon=ds.lon))
    grid_out = xr.Dataset(coords=dict(lat=ds_like.lat, lon=ds_like.lon))
    regridder = xe.Regridder(grid_in, grid_out, method=method)
    ds_regrid = regridder(ds)
    return ds_regrid


def soft_record_metric(hazard, da, da_obs, time_agg, metric="anom_std", theta_s=None):
    """Calculate the difference between two DataArrays."""
    dims = [d for d in da.dims if d not in ["lat", "lon"]]
    da_agg = da.reduce(time_agg_func, dim=dims)
    da_obs_agg = da_obs.reduce(time_agg_func, dim="time")

    # Regrid obs to model grid (after time aggregation)
    da_obs_agg_regrid = regrid_like(da_obs_agg, da_agg)
    anom = da_agg - da_obs_agg_regrid

    kwargs = dict(
        cmap=cmap_dict["anom"], ticks=hazard.ticks_anom, cbar_label=hazard.units_label
    )

    if metric == "anom_std":
        da_obs_std = da_obs.reduce(np.std, dim="time")
        da_obs_std_regrid = regrid_like(da_obs_std, da_agg)
        anom = anom / da_obs_std_regrid
        kwargs["cbar_label"] = f"{hazard.var_name}\n / standard deviation"

    elif metric == "anom_pct":
        anom = (anom / da_obs_agg_regrid) * 100
        kwargs["cbar_label"] = f"{hazard.var_name}\n anomaly [%]"
        kwargs["ticks"] = np.arange(-50, 51, 10)
        kwargs["cmap"] = cmap_dict["anom_b2r"]

    # elif metric == "ratio":
    #     anom = da_obs_agg_regrid / da_agg
    #     kwargs["cbar_label"] = "Percent [%]"
    #     kwargs["ticks"] = np.arange(-100, 101, 20)
    elif metric == "2000-year_ratio":
        # todo: 2000-yr model return period / obs_max
        rl = eva.get_return_level(
            2000, theta_s, dims=[d for d in da.dims if d not in ["lat", "lon"]][0]
        )
        anom = rl / da_obs_agg_regrid
        kwargs["cbar_label"] = f"2000-year return\n/ observed {time_agg}"
        kwargs["ticks"] = None

    return anom, kwargs


def plot_map_event_month_mode(hazard, ds, dataset):
    """Plot map of the most common month of hazard event."""
    ds["month_mode"] = (["lat", "lon"], mode(ds.event_time.dt.month, axis=0).mode)

    # Map of most common month
    fig, ax = plot_acs_hazard(
        data=ds["month_mode"],
        title=f"{hazard.name} most common month",
        date_range=date_range,
        cmap=plt.cm.gist_rainbow,
        cbar_extend="neither",
        ticks=np.arange(0.5, 12.5),
        tick_labels=list(calendar.month_name)[1:],
        cbar_label=None,
        dataset_name=dataset,
        outfile=f"{home}/figures/{hazard.index}_aus_month_mode_{dataset}.png",
        **plot_kwargs,
    )


def plot_map_event_year(
    hazard, ds, dataset, date_range, time_agg="maximum", time_dim="time"
):
    """Plot map of the year of the maximum or minimum event."""

    dt = ds[hazard.var].compute()
    dt.coords[time_dim] = dt.event_time.dt.year

    if hazard.time_agg == "maximum":
        da = dt.idxmax(time_dim)
    elif hazard.time_agg == "minimum":
        da = dt.idxmin(time_dim)

    # Map of year of maximum
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"Year of {metric} {hazard.name}",
        date_range=date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="neither",
        ticks=np.arange(1960, 2026, 5),
        tick_labels=None,
        cbar_label=None,
        dataset_name=dataset,
        outfile=f"{home}/figures/{hazard.index}_aus_{metric}_year_{dataset}.png",
        **plot_kwargs,
    )


def plot_map_time_agg(hazard, ds, dataset, date_range, time_agg="maximum"):
    """Plot map of time-aggregated hazard.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the hazard variable
    dataset : str
        Name of the dataset
    date_range : str
        Date range string
    hazard : Hazard
        Hazard information
    time_agg : str, default "maximum"
        Metric to aggregate over
    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[hazard.var].reduce(func_dict[time_agg], dim=dims)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{time_agg.capitalize()} {hazard.name}",
        date_range=date_range,
        cmap=hazard.cmap,
        cbar_extend="neither",
        ticks=hazard.ticks,
        tick_labels=None,
        cbar_label=hazard.units_label,
        dataset_name=dataset,
        outfile=f"{home}/figures/{hazard.index}_aus_{time_agg}_{dataset}.png",
        **plot_kwargs,
    )


def plot_map_obs_anom(
    hazard, ds, ds_obs, dataset, date_range, time_agg="maximum", metric="anom"
):
    """Plot map of time-aggregated hazard.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the hazard variable
    ds_obs : xr.Dataset
        Dataset containing the observational hazard variable
    dataset : str
        Name of the dataset
    date_range : str
        Date range string
    hazard : Hazard
        Hazard information
    time_agg : {"mean", "median", "max", "min", "sum"}, default "max"
        Time aggregation function name
    metric : {"anom", "anom_std", "anom_pct", "ratio"}, default "anom"
        Model/obs metric (see `soft_record_metric` for details)
    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[hazard.var].reduce(func_dict[time_agg], dim=dims)

    anom, kwargs = soft_record_metric(
        ds[hazard.var], ds_obs[hazard.var], hazard, time_agg, metric
    )

    fig, ax = plot_acs_hazard(
        data="anom_pct",
        title=f"{time_agg.capitalize()} {hazard.name} difference from observed",
        date_range=date_range,
        cbar_extend="both",
        tick_labels=None,
        dataset_name=f"{hazard.obs}, {dataset}",
        outfile=f"{home}/figures/{hazard.index}_aus_{metric[:3]}_{dataset}_obs_{metric}.png",
        **kwargs,
        **plot_kwargs,
    )


def get_gev_params(hazard, dataset, ds, covariate, core_dim="time", test="bic"):
    """Fit stationary and nonstationary GEV parameters to the hazard data.

    Parameters
    ----------
    hazard : Hazard
        Hazard information
    dataset : str
        Name of the dataset
    ds : xarray.Dataset
        Dataset containing the hazard variable
    covariate : xarray.DataArray
        Covariate for non-stationary GEV parameters
    core_dim : str, default "time"
        Core dimension for vectorized operations
    test : str, default "bic"
        Relative fit test to apply

    Returns
    -------
    theta_s : xarray.DataArray
        Stationary GEV parameters
    theta : xarray.DataArray
        Non-stationary GEV parameters
    """
    file_s = home / f"data/{hazard.index}_{dataset}_aus_params_stationary.nc"
    file_ns = home / f"data/{hazard.index}_{dataset}_aus_params_nonstationary_{test}.nc"

    # Stationary parameters
    if not file_s.exists():
        theta = eva.fit_gev(ds[hazard.var], stationary=True, core_dim=core_dim)
        theta = theta.to_dataset(name=hazard.var)
        theta.to_netcdf(file_s, compute=True)
    theta_s = xr.open_dataset(file_s)[hazard.var]

    # Non-stationary parameters
    if not file_ns.exists():
        theta = eva.fit_gev(
            ds[hazard.var],
            covariate=covariate,
            stationary=False,
            core_dim=core_dim,
            relative_fit_test=test,
        )
        theta = theta.to_dataset(name=hazard.var)
        theta.to_netcdf(file_ns, compute=True)
    theta = xr.open_dataset(file_ns)[hazard.var]
    return theta_s, theta


def plot_map_gev_param_trend(hazard, dataset, theta, date_range):
    """Plot map of GEV location and scale parameter trends."""
    for i, param in zip([2, 4], ["location", "scale"]):
        fig, ax = plot_acs_hazard(
            data=theta.isel(theta=i),
            title=f"TXx {param} parameter trend",
            date_range=date_range,
            cmap=cmap_dict["anom"],
            cbar_extend="both",
            ticks=hazard.ticks_param_trend[param],
            tick_labels=None,
            cbar_label=f"{param.capitalize()} trend\n[{hazard.units} / year]",
            dataset_name=dataset,
            outfile=home / f"figures/{hazard.index}_aus_{param}_trend_{dataset}.png",
            **plot_kwargs,
        )


def plot_map_return_level(hazard, dataset, theta, times, date_range, return_period=100):
    """Plot maps of return level for a given return period (times[0], times[1], times[1]-times[0])."""
    rl = eva.get_return_level(return_period, theta, times)

    for i, time in enumerate(times.values):
        fig, ax = plot_acs_hazard(
            data=rl.isel(time=i),
            title=f"1-in-{return_period} year {hazard.name}",
            date_range=time,
            baseline=date_range,
            cmap=hazard.cmap,
            cbar_extend="neither",
            ticks=hazard.ticks,
            tick_labels=None,
            cbar_label=hazard.units_label,
            dataset_name=dataset,
            outfile=f"{home}/figures/{hazard.index}_aus_return_level_{return_period}-year_{dataset}_{time}.png",
            **plot_kwargs,
        )

    # Time difference (i.e., change in return level)
    fig, ax = plot_acs_hazard(
        data=rl.isel(time=1, drop=True) - rl.isel(time=0, drop=True),
        title=f"1-in-{return_period} year {hazard.name} change",
        date_range=f"{times[1].item()}-{times[0].item()}",
        baseline=date_range,
        cmap=cmap_dict["anom"],
        cbar_extend="both",
        ticks=hazard.ticks_anom,
        tick_labels=None,
        cbar_label=hazard.units_label,
        dataset_name=dataset,
        outfile=f"{home}/figures/{hazard.index}_aus_return_level_{return_period}-year_{dataset}_{times[0].item()}-{times[1].item()}.png",
        **plot_kwargs,
    )


def plot_map_obs_return_period(
    hazard,
    dataset,
    ds,
    ds_obs,
    theta_s,
    date_range,
    time_dim="sample",
    time_agg="maximum",
):
    """Spatial map of return periods corresponding to the max/min value in obs."""
    da_agg = ds[hazard.var].reduce(func_dict[time_agg], time_dim=time_dim)
    da_obs_agg = ds_obs[hazard.var].reduce(func_dict[time_agg], dim="time")
    da_obs_agg_regrid = regrid_like(da_obs_agg, da_agg)

    rp = eva.get_return_period(da_obs_agg_regrid, theta_s, times)

    fig, ax = plot_acs_hazard(
        data=rp,
        title=f"Observed {hazard.name} {time_agg} {return_period}",
        date_range=date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="neither",
        # ticks=hazard.ticks,
        tick_labels=None,
        cbar_label="Return period [years]",
        dataset_name=f"{hazard.obs}, {dataset}",
        outfile=f"{home}/figures/{hazard.index}_aus_obs_{time_agg}_return_period_{dataset}.png",
        **plot_kwargs,
    )
    return


def plot_map_new_record_probability(
    hazard, dataset, ds, ds_obs, theta, theta_obs, date_range
):
    """Plot map of the probability of a new record event."""
    # todo
    return


if __name__ == "__main__":
    dataset = "MPI-ESM1-2-HR"
    # dataset = "AGCD"
    index = "txx"
    hazard = hazard_dict[index]
    ds_obs = get_obs_dataset(hazard)
    time_dim = "time"

    if dataset in dcpp_models:
        ds = get_dcpp_model_dataset(dataset, hazard)
        time_dim = "sample"
        ds = ds.stack(sample=["init_date", "ensemble", "lead_time"], create_index=False)
        ds = ds.transpose("sample", ...)
    elif dataset == hazard.obs:
        ds = ds_obs

    date_range = date_range_str(ds.time)

    plot_map_event_month_mode(hazard, ds, dataset)
    plot_map_event_year(hazard, ds, dataset, date_range, "maximum", time_dim)
    plot_map_time_agg(hazard, ds, dataset, date_range, "maximum")
    plot_map_time_agg(hazard, ds, dataset, date_range, "median")
    if dataset != hazard.obs:
        plot_map_obs_anom(hazard, ds, ds_obs, dataset, date_range, "median", "anom")
        plot_map_obs_anom(hazard, ds, ds_obs, dataset, date_range, "maximum", "anom")
        plot_map_obs_anom(
            hazard, ds, ds_obs, dataset, date_range, "maximum", "anom_std"
        )
        plot_map_obs_anom(
            hazard, ds, ds_obs, dataset, date_range, "maximum", "anom_pct"
        )

    # GEV analysis
    covariate = ds["time"].dt.year
    theta_s, theta = get_gev_params(hazard, dataset, ds, covariate, core_dim=time_dim)
    times = xr.DataArray([1962, 2020], dims="time")
    plot_map_gev_param_trend(hazard, dataset, theta, date_range)
    plot_map_return_level(hazard, dataset, theta, times, date_range, return_period=100)
    if dataset != hazard.obs:
        plot_map_obs_return_period(
            hazard,
            dataset,
            ds,
            ds_obs,
            theta_s,
            date_range,
            time_dim=time_dim,
            time_agg="maximum",
        )
