# -*- coding: utf-8 -*-


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

# from unseen.fileio import open_dataset, open_mfforecast
from unseen.spatial_selection import select_shapefile_regions
from unseen import independence, stability, eva, general_utils

sys.path.append("/g/data/xv83/as3189/plotting_maps/")
from acs_plotting_maps import plot_acs_hazard, regions_dict, cmap_dict


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


# @dataclass
# class Dataset:
#     name: str
#     long_name: float

#     def date_range(self):
#         return self.unit_price * self.quantity_on_hand
# Dataset = namedtuple("Dataset", "name long_name date_range ")

Hazard = namedtuple(
    "Hazard", "index name var units timescale obs cmap ticks ticks_anom"
)

hazard_dict = {
    "txx": Hazard(
        "txx",
        "TXx",
        "tasmax",
        units="Temperature [°C]",
        timescale="annual-jul-to-jun",
        obs="AGCD",
        cmap=cmap_dict["tasmax"],
        ticks=np.arange(20, 60 + 5, 5),
        ticks_anom=np.arange(-6, 7, 1),
    )
}
plot_kwargs = dict(
    name="aus_states_territories",
    regions=regions_dict["aus_states_territories"],
    mask_not_australia=True,
    figsize=[8 * 0.5, 6 * 0.5],
    area_linewidth=0.1,
    xlim=(110, 170),
    ylim=(-45, -5),
    label_states=False,
    contourf=False,
    contour=False,
    select_area=None,
    land_shadow=True,
    watermark=None,
)


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
    regridder = xe.Regridder(grid_in, grid_out, method)
    return regridder(ds)


def preprocess_dataset(ds):
    # Apply Australian land-sea mask
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    dims = [{d: 0} for d in ds.dims if d not in ["lat", "lon"]]
    mask = ds[ds.data_vars[0]].isel(dims)
    ds = select_shapefile_regions(ds, gdf, overlap_fraction=0.9)

    # Format event time as datetime
    if "event_time" in ds:
        ds["event_time"] = ds.event_time.astype(dtype="datetime64[ns]")
    return ds


def get_dcpp_model_dataset(dataset_name, hazard):
    file = list(
        home.rglob(f"data/{hazard.index}_{dataset_name}*_{hazard.timescale}_aus.nc")
    )
    dx = xr.open_dataset(str(file[0]), use_cftime=True, chunks="auto")
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


def plot_map_month_mode(ds, dataset_name, hazard):
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
        dataset_name=dataset_name,
        outfile=f"{home}/figures/{hazard.index}_map_month_mode_{dataset_name}.png",
        **plot_kwargs,
    )


def plot_map_event_year(
    ds, dataset_name, date_range, hazard, metric="maximum", time_dim="time"
):
    """Plot map of the year of the maximum or minimum event."""

    dt = ds[hazard.var].compute()
    dt.coords[time_dim] = dt.event_time.dt.year

    if metric == "maximum":
        da = dt.idxmax(time_dim)
    elif metric == "minimum":
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
        dataset_name=dataset_name,
        outfile=f"{home}/figures/{hazard.index}_map_{metric}_year_{dataset_name}.png",
        **plot_kwargs,
    )


def plot_time_agg_map(ds, ds_obs, dataset_name, date_range, hazard, metric="maximum"):
    """Plot map of time-aggregated hazard."""

    func_dict = {"median": np.median, "maximum": np.nanmax, "minimum": np.nanmin}
    func = func_dict[metric]
    dims = [
        d
        for d in ds.dims
        if d in ["time", "sample", "ensemble", "lead_time", "init_date"]
    ]

    da = ds[hazard.var].reduce(func, dim=dims)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{hazard.name} {metric}",
        date_range=date_range,
        cmap=hazard.cmap,
        cbar_extend="neither",
        ticks=hazard.ticks,
        tick_labels=None,
        cbar_label=hazard.units,
        dataset_name=dataset_name,
        outfile=f"{home}/figures/{hazard.index}_map_{metric}_{dataset_name}.png",
        **plot_kwargs,
    )

    # Model Obs Difference
    if ds_obs is not None:
        da_obs = ds_obs[hazard.var].reduce(func, dim="time")
        da_obs_regrid = regrid_like(da_obs, da)
        bias = da - da_obs_regrid
        fig, ax = plot_acs_hazard(
            data=bias,
            title=f"{metric} {index} difference from observations",
            date_range=date_range,
            cmap=cmap_dict["anom"],
            cbar_extend="both",
            ticks=hazard.ticks_anom,
            tick_labels=None,
            cbar_label=hazard.units,
            dataset_name=hazard.obs,
            outfile=f"{home}/figures/{hazard.index}_map_{metric}_{dataset_name}_obs_diff89.png",
            **plot_kwargs,
        )


def plot_eva():
    # EVA

    file_obs_params_s = file.parent / (file_obs.stem + "_params_stationary.nc")
    file_obs_params_ns = file.parent / (file_obs.stem + "_params_nonstationary.nc")
    file_params_s = file.parent / (file.stem + "_params_stationary.nc")
    file_params_ns = file.parent / (file.stem + "_params_nonstationary.nc")

    # Non-stationary parameters
    covariate_obs = ds_obs.time.dt.year
    if not file_obs_params_ns.exists():
        # ~ 14min
        theta_obs = eva.fit_gev(
            ds_obs[var], covariate=covariate_obs, stationary=False, core_dim="time"
        )
        theta_obs = theta_obs.to_dataset(name=var)

        theta_obs.to_netcdf(file_obs_params_ns, compute=True)
    theta_obs = xr.open_dataset(file_obs_params_ns)[var]
    # Stationary parameters (~7 mins)
    if not file_params_s.exists():
        theta_s = eva.fit_gev(ds_stacked[var], stationary=True, core_dim="sample")
        theta_s = theta_s.to_dataset(name=var)
        theta_s.to_netcdf(file_params_s, compute=True)
    theta_s = xr.open_dataset(file_params_s)[var]
    # Non-stationary parameters (~X mins)
    covariate = ds_stacked.time.dt.year
    if not file_params_ns.exists():
        theta = eva.fit_gev(
            ds_stacked[var], covariate=covariate, stationary=False, core_dim="sample"
        )
        theta = theta.to_dataset(name=var)
        theta.to_netcdf(file_params_ns, compute=True)
    theta = xr.open_dataset(file_params_ns)[var]
    # Plot location and scale parameter trends
    for j, param, ticks in zip(
        [2, 4],
        ["location", "scale"],
        [np.arange(-0.08, 0.09, 0.02), np.arange(-0.015, 0.016, 0.005)],
    ):
        for i, dx in enumerate([theta_obs, theta]):
            fig, ax = plot_acs_hazard(
                data=dx.isel(theta=j),
                title=f"TXx {param} parameter trend",
                date_range=date_range[i],
                cmap=cmap_dict["anom"],
                cbar_extend="both",
                ticks=ticks,
                tick_labels=None,
                cbar_label=f"{param.capitalize()} trend [°C / year]",
                dataset_name=dataset_name[i],
                outfile=home
                / f"figures/{index.lower()}_map_{param}_trend_{dataset_name[i]}.png",
                **plot_kwargs,
            )
            plt.show()

        # Return values
        return_period = 100
        times = xr.DataArray([1962, 2020], dims="time")

        rl = eva.get_return_level(return_period, theta, times)
        rl_obs = eva.get_return_level(return_period, theta_obs, times)

        for i, dx in enumerate([rl_obs, rl]):
            for j, time in enumerate(times.values):
                fig, ax = plot_acs_hazard(
                    data=dx.isel(time=j),
                    title=f"1-in-{return_period} year {index}",
                    date_range=time,
                    baseline=date_range[i],
                    cmap=cmap_dict["tasmax"],
                    cbar_extend="neither",
                    ticks=np.arange(20, 60 + 5, 5),
                    tick_labels=None,
                    cbar_label=units,
                    dataset_name=dataset_name[i],
                    outfile=home
                    / f"figures/{index.lower()}_map_{return_period}-year_return_level_{dataset_name[i]}_{time}.png",
                    **plot_kwargs,
                )
                plt.show()

            # Difference
            time_diff = f"{times[1].item()}-{times[0].item()}"
            fig, ax = plot_acs_hazard(
                data=dx.isel(time=1, drop=True) - dx.isel(time=0, drop=True),
                title=f"1-in-{return_period} year {index} change",
                date_range=time_diff,
                baseline=date_range[1],
                cmap=cmap_dict["anom"],
                cbar_extend="both",
                ticks=np.arange(-6, 7, 1),
                tick_labels=None,
                cbar_label=units,
                dataset_name=dataset_name[i],
                outfile=home
                / f"figures/{index.lower()}_map_return_level_{return_period}-year_{dataset_name[i]}_{time_diff}.png",
                **plot_kwargs,
            )
            plt.show()


if __name__ == "__main__":
    dataset_name = "MPI-ESM1-2-HR"
    index = "txx"
    hazard = hazard_dict[index]
    ds_obs = get_obs_dataset(hazard)
    time_dim = "time"

    if dataset_name in dcpp_models:
        ds = get_dcpp_model_dataset(dataset_name, hazard)
        time_dim = "sample"
        ds = ds.stack(sample=["init_date", "ensemble", "lead_time"], create_index=False)
        ds = ds.transpose("sample", ...)

    date_range = date_range_str(ds.time)

    plot_map_month_mode(ds, dataset_name, hazard)
    plot_map_event_year(ds, dataset_name, date_range, hazard, "maximum", time_dim)
    plot_time_agg_map(ds, ds_obs, dataset_name, date_range, hazard, metric="maximum")
    plot_time_agg_map(ds, ds_obs, dataset_name, date_range, hazard, metric="minimum")
