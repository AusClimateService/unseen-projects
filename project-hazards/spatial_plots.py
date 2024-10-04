# -*- coding: utf-8 -*-
"""UNSEEN analysis spatial maps of climate hazards in Australian."""

import argparse
import calendar
from collections import namedtuple
import geopandas as gp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
from scipy.stats import genextreme, mode
import xarray as xr

from unseen import (
    fileio,
    eva,
    spatial_selection,
    general_utils,
    independence,
    similarity,
)
from acs_plotting_maps import plot_acs_hazard, cmap_dict, tick_dict
from cfg import aep_to_ari

np.set_printoptions(suppress=True)

home = Path("/g/data/xv83/unseen-projects/outputs/hazards")

plot_kwargs = dict(
    name="aus_states_territories",
    mask_not_australia=True,
    figsize=[8 * 0.8, 6 * 0.8],
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
    "median": np.nanmedian,
    "maximum": np.nanmax,
    "minimum": np.nanmin,
    "sum": np.sum,
}

Hazard = namedtuple(
    "Hazard",
    "idx index var var_name units units_label timescale obs_name cmap ticks ticks_anom ticks_param_trend",
)

hazard_dict = {
    "txx": Hazard(
        idx="txx",
        index="TXx",
        var="tasmax",
        var_name="Temperature",
        units="°C",
        units_label="Temperature [°C]",
        timescale="annual-jul-to-jun",
        obs_name="AGCD",
        cmap=cmap_dict["tasmax"],
        ticks=np.arange(20, 60 + 5, 5),
        ticks_anom=tick_dict["tas_anom_mon"],
        ticks_param_trend={
            "location": np.arange(-1.6, 1.7, 0.2),  # todo: update
            "scale": np.arange(-0.3, 0.35, 0.05),  # todo: update
        },
    )
}


class Dataset:
    """Class to store dataset information and get forecast and observations.

    Parameters
    ----------
    name : str
        Dataset name
    index : str
        Hazard index
    fcst_file : Path
        Forecast data file path
    obs_file : Path
        Observational data file path
    bias_correction : str, default None
        Bias correction method

    Attributes
    ----------
    name : str
        Dataset name
    index : str
        Hazard index
    fcst_file : Path
        Forecast file path
    filename : str
        Forecast file name
    obs_file : Path
        Observational file path
    bias_correction : str, default None
        Bias correction method
    fig_dir : Path
        Figure output directory
    hazard : Hazard
        Hazard information
    time_dim : str
        Time dimension name (e.g., "sample" for large ensemble)
    long_name : str
        Dataset long name (i.e., MIROC6 additive bias corrected large ensemble)
    date_range : str
        Date range of dataset (e.g., "01 January 1960 to 31 December 2024")
    date_range_obs : str
        Date range of observational dataset

    """

    def __init__(
        self,
        name,
        index,
        fcst_file,
        obs_file,
        bias_correction=None,
    ):
        """Initialise Dataset instance."""
        self.name = name
        self.index = index
        self.fcst_file = fcst_file
        self.filename = self.fcst_file.name
        self.obs_file = obs_file
        self.bias_correction = bias_correction
        self.fig_dir = f"{home}/figures/{self.index}/"

        # Get variables from hazard_dict
        self.hazard = hazard_dict[index]
        for key, value in self.hazard._asdict().items():
            setattr(self, key, value)

        if self.is_obs():
            self.time_dim = "time"
            # N.BB. Add extra space at end (avoids cut off in plots)
            self.long_name = self.name + " "

        else:
            self.time_dim = "sample"
            self.long_name = f"{self.name} large ensemble "
            if self.bias_correction:
                self.long_name += f"({self.bias_correction} bias corrected; samples=X) "
            else:
                self.long_name += "(samples=X) "

    def is_obs(self):
        """Check if dataset is observational."""
        return self.name == self.obs_name

    def date_range_str(self, time):
        """Return date range 'DD month YYYY' string from time coordinate."""
        # Note that this assumes annual data & time indexed by YEAR_END_MONTH
        if time.ndim > 1:
            # Stack time dimension to get min and max
            time = time.stack(time=time.dims)

        # First and last year
        year = [f(time.dt.year.values) for f in [np.min, np.max]]

        # Index of year end month
        year_end_month = time.dt.month[0].item()
        if year_end_month != 12:
            # Times based on end month of year, so previous year is the start
            year[0] -= 1
        YE_ind = [year_end_month + i for i in [1, 0]]
        # Adjust for December (convert 13 to 1)
        YE_ind[1] = 1 if YE_ind[1] == 13 else YE_ind[1]

        # First and last month name
        mon = [list(calendar.month_name)[i] for i in YE_ind]

        day = [1, calendar.monthrange(year[1], YE_ind[1])[-1]]
        date_range = " to ".join([f"{day[i]} {mon[i]} {year[i]}" for i in [0, 1]])
        return date_range

    def __str__(self):
        """Return string representation of Dataset instance."""
        return f"{self.name}"

    def __repr__(self):
        """Return string/dataset representation of Dataset instance."""
        if hasattr(self, "ds"):
            return self.ds.__repr__()
        else:
            return self.name


def get_dataset(
    filename,
    start_year=None,
    mask_ocean=False,
    overlap_fraction=0.1,
    min_lead=None,
    min_lead_kwargs={},
    similarity=False,
    alpha=0.05,
):
    """Get index model dataset.

    Parameters
    ----------
    filename : str
        File path to model data
    start_year : int, optional
        Start year
    mask_not_australia : bool, default True
        Apply Australian land-sea mask
    overlap_fraction : float, default 0.1
        Fraction of overlap required for shapefile selection
    min_lead : str or int, optional
        Minimum lead time (file path or integer)
    min_lead_kwargs : dict, optional
        Minimum lead time fileio.open_dataset keyword arguments
    similarity : str, optional
        Similarity mask file path
    alpha : float, default 0.05
        Significance level for AD test

    Returns
    -------
    ds : xarray.Dataset
        Model dataset
    """

    ds = xr.open_dataset(str(filename), use_cftime=True)

    if start_year:
        # Drop years before model data is available
        ds = ds.sel(time=slice(str(start_year), None))

    if "event_time" in ds:
        # Format event time as datetime
        ds["event_time"] = ds.event_time.astype(dtype="datetime64[ns]")

    if min_lead and "lead_time" in ds:
        # Drop lead times less than min_lead
        if isinstance(min_lead, int):
            ds = ds.where(ds["lead_time"] >= min_lead)
        else:
            # Load min_lead from file
            ds_min_lead = fileio.open_dataset(str(min_lead), **min_lead_kwargs)
            min_lead = ds_min_lead["min_lead"].load()
            ds = ds.groupby("init_date.month").where(ds["lead_time"] >= min_lead)
            ds = ds.drop_vars("month")
            ds["min_lead"] = min_lead
        ds = ds.dropna(dim="lead_time", how="all")

    if mask_ocean:
        # Apply Australian land-sea mask
        ds = mask_not_australia(ds, overlap_fraction=overlap_fraction)

    if similarity:
        # Apply similarity mask
        ds_gof = fileio.open_dataset(str(similarity))
        ds = ds.where(ds_gof["pval"] > 0.05)

    if all([dim in ds.dims for dim in ["init_date", "ensemble", "lead_time"]]):
        # Stack sample dimensions
        ds = ds.stack(sample=["init_date", "ensemble", "lead_time"], create_index=False)
        ds = ds.transpose("sample", ...)
    return ds


def mask_not_australia(ds, overlap_fraction=0.1):
    """Apply Australian land-sea mask to dataset."""
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    ds = spatial_selection.select_shapefile_regions(
        ds, gdf, overlap_fraction=overlap_fraction
    )
    return ds


def soft_record_metric(data, da, da_obs, time_agg, metric="anom_std", theta=None):
    """Calculate the difference between two DataArrays.

    Parameters
    ----------
    data : Dataset
        Dataset information
    da : xarray.DataArray
        Model data
    da_obs : xarray.DataArray
        Observational data
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    metric : {"anom", "anom_std", "anom_pct", "anom_2000yr"}, default "anom"
        Model/obs metric (see below)
    theta_s : xarray.DataArray, optional
        Nonstationary GEV parameters # todo: change to ns

    Returns
    -------
    anom : xarray.DataArray
        Difference between model and obs
    kwargs : dict
        Plotting keyword arguments
    """
    dims = [d for d in da.dims if d not in ["lat", "lon"]]
    da_agg = da.reduce(func_dict[time_agg], dim=dims)
    da_obs_agg = da_obs.reduce(func_dict[time_agg], dim="time")
    # todo: change scale to centre white
    # todo: add subsampled version
    # Regrid obs to model grid (after time aggregation)
    da_obs_agg_regrid = general_utils.regrid(da_obs_agg, da_agg)
    anom = da_agg - da_obs_agg_regrid

    kwargs = dict(
        cmap=cmap_dict["anom"],
        ticks=data.ticks_anom,
        cbar_label=data.units_label,
        cbar_extend="both",
    )

    if metric == "anom_std":
        da_obs_std = da_obs.reduce(np.std, dim="time")
        da_obs_std_regrid = general_utils.regrid(da_obs_std, da_agg)
        anom = anom / da_obs_std_regrid
        kwargs["cbar_label"] = f"Obs. standard\ndeviation"  # todo: update/check

    elif metric == "anom_pct":
        anom = (anom / da_obs_agg_regrid) * 100
        kwargs["cbar_label"] = f"Difference [%]"
        kwargs["ticks"] = np.arange(-50, 51, 10)

    elif metric == "anom_2000yr":
        rl = eva.get_return_level(2000, theta, dims=dims)
        anom = rl / da_obs_agg_regrid
        kwargs["cbar_label"] = f"2000-year return\n/ {time_agg[:3]}(obs)"
        kwargs["ticks"] = np.arange(0, 2.2, 0.2)
        kwargs["cbar_extend"] = "neither"
    return anom, kwargs


def get_gev_params(data, ds, covariate, test="bic", fitstart="LMM"):
    """Fit stationary and nonstationary GEV parameters to the hazard data.

    Parameters
    ----------
    data : Dataset
        Dataset information
    ds : xarray.Dataset
        Dataset containing the hazard variable
    covariate : xarray.DataArray
        Covariate for non-stationary GEV parameters
    test : str, default "bic"
        Relative fit test to apply
    fitstart : str, default "LMM"
        Fit start method

    Returns
    -------
    theta_s : xarray.DataArray
        Stationary GEV parameters
    theta : xarray.DataArray
        Non-stationary GEV parameters
    """
    file_s = home / f"data/GEV_stationary_params_{data.filename}.nc"
    file_ns = home / f"data/GEV_nonstationary_params_{test}_{data.filename}.nc"

    # Stationary parameters
    if not file_s.exists():
        theta_s = eva.fit_gev(ds[data.var], fitstart=fitstart, core_dim=data.time_dim)
        theta_s = theta_s.to_dataset(name=data.var)
        theta_s.to_netcdf(file_s, compute=True)
    theta_s = xr.open_dataset(file_s)[data.var]

    # Non-stationary parameters
    if not file_ns.exists():
        theta_ns = eva.fit_gev(
            ds[data.var],
            covariate=covariate,
            stationary=False,
            core_dim=data.time_dim,
            fitstart=fitstart,
            relative_fit_test=test,
        )
        theta_ns = theta_ns.to_dataset(name=data.var)
        theta_ns.to_netcdf(file_ns, compute=True)
    theta_ns = xr.open_dataset(file_ns)[data.var]
    return theta_s, theta_ns


def plot_map_event_month_mode(data, ds):
    """Plot map of the most common month of hazard event.

    Parameters
    ----------
    data : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Dataset containing the hazard variable
    """
    da = xr.DataArray(
        mode(ds.event_time.dt.month, axis=0).mode,
        coords=dict(lat=ds.lat, lon=ds.lon),
        dims=["lat", "lon"],
    )

    # Map of most common month
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{data.index} most common month",
        date_range=data.date_range,
        cmap=plt.cm.gist_rainbow,
        cbar_extend="neither",
        ticks=np.arange(0.5, 12.5),
        tick_labels=list(calendar.month_name)[1:],
        cbar_label=None,
        dataset_name=data.long_name,
        outfile=f"{data.fig_dir}/{data.idx}_aus_month_mode_{data.filename}.png",
        **plot_kwargs,
    )


def plot_map_event_year(data, ds, time_agg="maximum"):
    """Plot map of the year of the maximum or minimum event.

    Parameters
    ----------
    data : Dataset
        Dataset information
    ds : xarray.Dataset
        Dataset containing the hazard variable
    time_agg : {"maximum", "minimum"}, default "maximum"
        Time aggregation function name
    """

    dt = ds[data.var].copy().compute()
    dt.coords[data.time_dim] = dt.event_time.dt.year

    if time_agg == "maximum":
        da = dt.idxmax(data.time_dim)
    elif time_agg == "minimum":
        da = dt.idxmin(data.time_dim)

    # Map of year of maximum
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"Year of {time_agg} {data.index}",
        date_range=data.date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="max",
        ticks=np.arange(1960, 2026, 5),
        tick_labels=None,
        cbar_label=None,
        dataset_name=data.long_name,
        outfile=f"{data.fig_dir}/year_{time_agg}_{data.filename}.png",
        **plot_kwargs,
    )


def plot_map_time_agg(data, ds, time_agg="maximum"):
    """Plot map of time-aggregated data.

    Parameters
    ----------
    data : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Dataset containing the hazard variable
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[data.var].reduce(func_dict[time_agg], dim=dims)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{time_agg.capitalize()} {data.index}",
        date_range=data.date_range,
        cmap=data.cmap,
        cbar_extend="neither",
        ticks=data.ticks,
        tick_labels=None,
        cbar_label=data.units_label,
        dataset_name=data.long_name,
        outfile=f"{data.fig_dir}/{time_agg}_{data.filename}.png",
        **plot_kwargs,
    )


def plot_map_obs_anom(
    data,
    ds,
    ds_obs,
    time_agg="maximum",
    metric="anom",
    theta=None,
):
    """Plot map of time-aggregated data.

    Parameters
    ----------
    data : Dataset
        Dataset information
    ds : xr.Dataset
        Dataset containing the hazard variable
    ds_obs : xr.Dataset
        Dataset containing the observational hazard variable
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    metric : {"anom", "anom_std", "anom_pct", "ratio"}, default "anom"
        Model/obs metric (see `soft_record_metric` for details)
    theta : xarray.DataArray, optional
        Non-stationary GEV parameters #todo: define covariate year

    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[data.var].reduce(func_dict[time_agg], dim=dims)

    anom, kwargs = soft_record_metric(
        data, ds[data.var], ds_obs[data.var], time_agg, metric, theta
    )

    fig, ax = plot_acs_hazard(
        data=anom,
        title=f"{time_agg.capitalize()} {data.index} \ndifference from observed",
        date_range=data.date_range_obs,
        tick_labels=None,
        dataset_name=f"{data.obs_name}, {data.long_name}",
        outfile=f"{data.fig_dir}/{time_agg}_{metric}_{data.filename}.png",
        **kwargs,
        **plot_kwargs,
    )


def plot_map_gev_param_trend(data, theta, covariate):
    """Plot map of GEV location and scale parameter trends.

    Parameters
    ----------
    data : Dataset
        Dataset information instance
    theta : xarray.Dataset
        Non-stationary GEV parameters
    covariate : xarray.DataArray
        Covariate for non-stationary GEV parameters
    """
    _, loc, scale = eva.unpack_gev_params(theta, covariate)
    for param, name in zip([loc, scale], ["location", "scale"]):

        fig, ax = plot_acs_hazard(
            data=param.isel({data.time_dim: -1})
            - param.isel({data.time_dim: 0}),  # todo: convert to deg/year or deg/decade
            title=f"{data.index} {name} parameter trend",
            date_range=data.date_range,
            cmap=cmap_dict["anom"],
            cbar_extend="both",
            ticks=data.ticks_param_trend[name],
            # tick_labels=data.ticks_param_trend[name][::2],
            cbar_label=f"{name.capitalize()} trend",
            dataset_name=data.long_name,
            outfile=f"{data.fig_dir}/GEV_{name}_trend_{data.filename}.png",
            **plot_kwargs,
        )


def plot_map_aep(data, theta, times, aep=1):
    """Plot maps of AEP for a given threshold.

    Parameters
    ----------
    data : Dataset
        Dataset information instance
    theta : xarray.Dataset
        Non-stationary GEV parameters
    times : xarray.DataArray
        Start and end years for AEP calculation
    aep : int, default 1
        Annual exceedance probability threshold

    Notes
    -----
       - AEP = 1 / RL
       - Plot AEP for times[0], times[1] and the difference between the two.
    """
    # todo convert to AEP
    ari = aep_to_ari(aep)

    da_aep = eva.get_return_level(ari, theta, times)
    # da_aep = (1 / rl) * 100
    # da_aep = ari_to_aep(da_ari)

    for i, time in enumerate(times.values):
        fig, ax = plot_acs_hazard(
            data=da_aep.isel({data.time_dim: i}),
            title=f"{data.index} {aep}% Annual Exceedance Probability",
            date_range=time,
            cmap=data.cmap,
            cbar_extend="neither",
            ticks=data.ticks,
            tick_labels=None,
            cbar_label=data.units_label,
            dataset_name=data.long_name,
            outfile=f"{data.fig_dir}/AEP_{int(aep)}-percent_{data.filename}_{time}.png",
            **plot_kwargs,
        )

    # Time difference (i.e., change in return level)
    fig, ax = plot_acs_hazard(
        data=da_aep.isel({data.time_dim: -1}, drop=True)
        - da_aep.isel({data.time_dim: 0}, drop=True),
        title=f"{data.index} {aep}% Annual Exceedance Probability change",
        date_range=f"{times[1].item()}-{times[0].item()}",
        cmap=cmap_dict["anom"],
        cbar_extend="both",
        ticks=data.ticks_anom,  # todo: reduce scale
        tick_labels=None,
        cbar_label=data.units_label,
        dataset_name=data.long_name,
        outfile=f"{home}/figures/AEP_{int(aep)}-percent_{data.filename}_{times[0].item()}-{times[1].item()}.png",
        **plot_kwargs,
    )


def plot_map_obs_ari(
    data,
    ds,
    ds_obs,
    theta,
    time_agg="maximum",
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    data : Dataset
        Dataset information
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    theta : xarray.DataArray
        Non-stationary GEV parameters
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    """
    da_agg = ds[data.var].reduce(func_dict[time_agg], dim=data.time_dim)
    da_obs_agg = ds_obs[data.var].reduce(func_dict[time_agg], dim="time")
    da_obs_agg_regrid = general_utils.regrid(da_obs_agg, da_agg)

    rp = xr.apply_ufunc(
        eva.get_return_period,
        da_obs_agg_regrid,
        theta,
        input_core_dims=[[], ["theta"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )

    fig, ax = plot_acs_hazard(
        data=rp,
        title=f"UNSEEN return period of\nobserved {data.index} {time_agg}",
        date_range=data.date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="max",
        norm=LogNorm(vmin=1, vmax=10000),
        cbar_label=f"Modelled return\nperiod [years]",
        dataset_name=f"{data.obs_name}, {data.long_name}",
        outfile=f"{data.fig_dir}/ARI_obs_{time_agg}_{data.filename}.png",
        **plot_kwargs,
    )
    return


def plot_map_new_record_probability(data, ds, ds_obs, theta, time_agg, ari=10):
    """Plot map of the probability of breaking the obs record in the next X years.

    Parameters
    ----------
    data : Dataset
        Dataset information
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    theta : xarray.DataArray
        Non-stationary GEV parameters
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    ari : int, default 10
        Return period in years
    """

    def new_record_probability(record, theta, ari):
        """Probability of exceeding a record in the next {ari} years."""
        shape, loc, scale = eva.unpack_gev_params(theta)
        # Probability of exceeding the record in a single year
        annual_probability = 1 - genextreme.cdf(record, shape, loc=loc, scale=scale)
        # Probability of exceeding the record at least once over the specified period
        cumulative_probability = 1 - (1 - annual_probability) ** ari
        # Convert to percentage
        probability = cumulative_probability * 100
        return probability

    da = ds[data.var].reduce(func_dict[time_agg], dim=data.time_dim)
    if data.name != data.obs_name:
        da_obs_agg = ds_obs[data.var].reduce(func_dict[time_agg], dim="time")
        da = general_utils.regrid(da_obs_agg, da)

    rl, probability = xr.apply_ufunc(
        new_record_probability,
        da,
        theta,
        input_core_dims=[[], ["theta"]],
        output_core_dims=[[], []],
        kwargs=dict(return_period=ari),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"] * 2,
    )

    fig, ax = plot_acs_hazard(
        data=probability,
        title=f"Probability of record breaking \n{data.index} in the next {ari} years",
        date_range=data.date_range,
        cmap=cmap_dict["anom"],
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label=f"Probability [%]",
        dataset_name=(
            data.obs_name if data.is_obs() else f"{data.obs_name}, {data.long_name}"
        ),
        outfile=f"{data.fig_dir}/new_record_probability_{ari}-year_{data.filename}.png",
        **plot_kwargs,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("dataset", type=str, help="Variable name")
    # parser.add_argument("index", type=str, help="Data files to regrid")
    # parser.add_argument("fcst_file", type=str, help="Forecast data file")
    # parser.add_argument("obs_file", type=str, help="Observational data file")
    # parser.add_argument("start_year", type=int, help="Start year")
    # parser.add_argument("bias_correction", type=str, default=None, help="Bias correction method")
    # parser.add_argument("mask_similarity", type=bool, default=True, help="Apply similarity mask")
    # parser.add_argument("min_lead", type=str, help="Minimum lead time file")
    # parser.add_argument("min_lead_kwargs", type=dict, help="Minimum lead time file")
    # parser.add_argument("time_agg", type=str, default="maximum", help="Time aggregation method")
    # args = parser.parse_args()

    dataset = "MIROC6"  # "CanESM5"
    # dataset = "AGCD"
    index = "txx"
    bias_correction = "additive"
    similarity_mask = True  # todo: add mask_similarity
    overlap_fraction = 0.1
    time_agg = "maximum"

    # Filenames
    obs_file = home / "data/txx_AGCD-CSIRO_r05_1901-2024_annual-jul-to-jun_aus.nc"
    fcst_file = f"{index}_{dataset}*_aus.nc"
    if bias_correction is not None:
        fcst_file = fcst_file[:-3] + f"*{bias_correction}.nc"
    fcst_file = list(Path(f"{home}/data/").rglob(fcst_file))[0]
    similarity_file = home / f"data/similarity-test_{fcst_file.name}"
    min_lead = list(
        Path(f"{home}/data/").rglob(f"independence-test_{index}_{dataset}*aus.nc")
    )[0]

    min_lead_kwargs = dict(
        shapefile=f"{home}/shapefiles/australia.shp",
        shape_overlap=0.1,
        spatial_agg="median",
    )

    # Load data
    data = Dataset(
        dataset,
        index,
        fcst_file,
        obs_file,
        bias_correction=bias_correction,
    )
    ds = data.get_dataset(
        fcst_file,
        mask_not_australia=False,
        min_lead=min_lead,
        min_lead_kwargs=min_lead_kwargs,
    )
    if not data.is_obs():
        # Add sample size to model long_name
        n_samples = ds[data.var].dropna(data.time_dim, how="any")[data.time_dim].size
        data.long_name.replace("samples=X", f"samples={n_samples}")
        start_year = ds.time.dt.year.min().item()
    else:
        start_year = 1961  # End of first year

    ds_obs = data.get_dataset(
        obs_file,
        start_year=start_year,
        mask_not_australia=False,
    )

    ds_ind = xr.open_dataset(str(min_lead), use_cftime=True)
    ds_similarity = xr.open_dataset(str(similarity_file), use_cftime=True)
    similarity.similarity_spatial_plot(
        ds_similarity,
        dataset,
        outfile=f"{data.fig_dir}/{similarity_file.name[:-3]}.png",
    )
    independence.spatial_plot(
        ds_ind, suptitle=dataset, outfile=f"{data.fig_dir}/{min_lead.name[:-3]}.png"
    )

    plot_map_event_month_mode(data, ds)
    plot_map_event_year(data, ds, time_agg)
    plot_map_time_agg(data, ds, "median")
    plot_map_time_agg(data, ds, time_agg)

    if not data.is_obs():
        plot_map_obs_anom(data, ds, ds_obs, "median", "anom")
        plot_map_obs_anom(data, ds, ds_obs, time_agg, "anom")
        plot_map_obs_anom(data, ds, ds_obs, time_agg, "anom_std")
        plot_map_obs_anom(data, ds, ds_obs, time_agg, "anom_pct")

    # GEV analysis
    covariate = ds["time"].dt.year
    theta_s, theta = get_gev_params(data, ds, covariate, test="bic")
    times = xr.DataArray([start_year, 2020], dims=data.time_dim)
    plot_map_gev_param_trend(data, theta, times)
    plot_map_aep(data, theta, times, 1)
    plot_map_new_record_probability(
        data,
        ds,
        ds_obs,
        theta,
        data.date_range_obs,
        time_agg,
        ari=10,
    )
    if not data.is_obs():
        plot_map_obs_anom(data, ds, ds_obs, time_agg, "anom_2000yr", theta_s=theta_s)
        plot_map_obs_ari(
            data,
            ds,
            ds_obs,
            theta_s,
            time_agg=time_agg,
        )
    else:
        plot_map_obs_ari(
            data,
            ds,
            ds_obs,
            theta,
            covariate=xr.DataArray([2024 + 5], dims="time"),
            time_agg=time_agg,
        )
