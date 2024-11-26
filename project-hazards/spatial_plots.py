# -*- coding: utf-8 -*-
"""UNSEEN analysis spatial maps of climate hazards in Australian."""

import argparse
import calendar
import geopandas as gp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
from scipy.stats import genextreme, mode
import xarray as xr

from unseen import eva, general_utils
from acs_plotting_maps import plot_acs_hazard, cmap_dict, tick_dict


plot_kwargs = dict(
    name="ncra_regions",
    mask_not_australia=True,
    figsize=[8, 6],
    xlim=(114, 162),
    ylim=(-43, -8.5),
    contourf=False,
    contour=False,
    select_area=None,
    land_shadow=False,
    watermark=None,
)

func_dict = {
    "mean": np.mean,
    "median": np.nanmedian,
    "maximum": np.nanmax,
    "minimum": np.nanmin,
    "sum": np.sum,
}


class InfoSet:
    """Repository of dataset information to pass to plot functions.

    Parameters
    ----------
    name : str
        Dataset name
    metric : str
        Metric/index variable (lowercase; modified by `kwargs`)
    file : Path
        Forecast file path
    ds : xarray.Dataset, optional
        Model or observational dataset
    ds_obs : xarray.Dataset, optional
        Observational dataset (only if different from ds)
    bias_correction : str, default None
        Bias correction method
    fig_dir : Path
        Figure output directory
    date_dim : str
        Time dimension name for date range (e.g., "sample" or "time")
    kwargs : dict
        Additional metric-specific attributes (idx, var, var_name, units, units_label, freq, obs_name, cmap, cmap_anom, ticks, ticks_anom, ticks_param_trend)

    Attributes
    ----------
    name : str
        Dataset name
    file : str or pathlib.Path
        File path of model or observational metric dataset
    bias_correction : str, default None
        Bias correction method
    fig_dir : str or pathlib.Path, optional
        Figure output directory. Default is the user's home directory.
    date_range : str
        Date range string
    date_range_obs : str
        Date range string for observational dataset
    time_dim : str
        Time dimension name (e.g., "sample" or "time")
    long_name : str
        Dataset long name (e.g., "ACCESS-CM2 ensemble")
    long_name_with_obs : str
        Dataset long name with observational dataset (e.g., "AGCD, ACCESS-CM2 ensemble")

    Functions
    ---------
    filestem(mask=False)
        Return filestem with or without "_masked" suffix
    is_model()
        Check if dataset is a model

    Notes
    -----
    * Includes all variables from `kwargs`
    """

    def __init__(
        self,
        name,
        metric,
        file,
        ds=None,
        ds_obs=None,
        bias_correction=None,
        fig_dir=Path.home(),
        date_dim="time",
        **kwargs,
    ):
        """Initialise Dataset instance."""
        self.name = name
        self.metric = metric
        self.file = Path(file)
        self.bias_correction = bias_correction
        self.fig_dir = Path(fig_dir)

        # Get variables from hazard_dict
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.cmap_anom.set_bad("lightgrey")
        self.cmap.set_bad("lightgrey")

        # Set dataset-specific attributes
        if ds is not None:
            self.date_range = date_range_str(ds[date_dim], self.freq)
        if ds_obs is not None:
            self.date_range_obs = date_range_str(ds_obs.time, self.freq)

        if self.is_model():
            self.time_dim = "sample"
            self.long_name = f"{self.name} ensemble"
            if self.bias_correction:
                self.long_name += f" ({self.bias_correction} bias corrected)"
            # else:
            #     self.n_samples = ds[self.var].dropna("sample", how="any")["sample"].size
            #     self.long_name += f"(samples={self.n_samples})"
            self.long_name_with_obs = f"{self.obs_name}, {self.long_name}"
        else:
            self.time_dim = "time"
            self.long_name = f"{self.name}"
            self.long_name_with_obs = self.long_name

    def filestem(self, mask=False):
        """Return filestem with or without "_masked" suffix."""
        stem = self.file.stem
        if mask is not None:
            stem += "_masked"
        return stem

    def is_model(self):
        """Check if dataset is a model."""
        return self.name != self.obs_name

    def __str__(self):
        """Return string representation of Dataset instance."""
        return f"{self.name}"

    def __repr__(self):
        """Return string/dataset representation of Dataset instance."""
        if hasattr(self, "ds"):
            return self.ds.__repr__()
        else:
            return self.name


def date_range_str(time, freq=None):
    """Return date range 'DD month YYYY' string from time coordinate.

    Parameters
    ----------
    time : xarray.DataArray
        Time coordinate
    freq : str, optional
        Frequency string (e.g., "YE-JUN")
    """

    # Note that this assumes annual data & time indexed by YEAR_END_MONTH
    if time.ndim > 1:
        # Stack time dimension to get min and max
        time = time.stack(time=time.dims)

    # First and last year
    year = [f(time.dt.year.values) for f in [np.min, np.max]]

    # Index of year end month
    if freq:
        # Infer year end month from frequency string
        year_end_month = list(calendar.month_abbr).index(freq[-3:].title())
    else:
        # Infer year end month from time coordinate
        year_end_month = time.dt.month[0].item()

    if year_end_month != 12:
        # Times based on end month of year, so previous year is the start
        year[0] -= 1  # todo: Add check for freq str starting with "YE"
    YE_ind = [year_end_month + i for i in [1, 0]]
    # Adjust for December (convert 13 to 1)
    YE_ind[1] = 1 if YE_ind[1] == 13 else YE_ind[1]

    # First and last month name
    mon = [list(calendar.month_name)[i] for i in YE_ind]

    day = [1, calendar.monthrange(year[1], YE_ind[1])[-1]]
    date_range = " to ".join([f"{day[i]} {mon[i]} {year[i]}" for i in [0, 1]])
    return date_range


def plot_time_agg(info, ds, time_agg="maximum", mask=None, savefig=True):
    """Plot map of time-aggregated data.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model or observational dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    mask : xarray.DataArray, default None
        Apply model similarity mask
    savefig : bool, default True
        Save figure to file
    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[info.var].reduce(func_dict[time_agg], dim=dims)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{time_agg.capitalize()} {info.metric}",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend="both",
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        stippling=mask,
        outfile=f"{info.fig_dir}/{time_agg}_{info.filestem(mask)}.png",
        savefig=savefig,
        **plot_kwargs,
    )


def plot_time_agg_subsampled(info, ds, ds_obs, time_agg="maximum", resamples=1000):
    """Plot map of obs-sized subsample of data (sample median of time-aggregate).

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    resamples : int, default 1000
        Number of random samples of subsampled data
    # mask : xarray.DataArray, default None
    #     Show model similarity stippling mask
    """
    assert "pval_mask" in ds.data_vars, "Model similarity mask not found in dataset."

    rng = np.random.default_rng(seed=0)
    n_obs_samples = ds_obs[info.var].time.size

    def rng_choice_resamples(data, size, resamples):
        """Return resamples of size samples from data."""
        return np.stack(
            [rng.choice(data, size=size, replace=False) for _ in range(resamples)]
        )

    da_subsampled = xr.apply_ufunc(
        rng_choice_resamples,
        ds[info.var],
        input_core_dims=[[info.time_dim]],
        output_core_dims=[["k", "subsample"]],
        kwargs=dict(size=n_obs_samples, resamples=resamples),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs=dict(
            output_sizes=dict(k=resamples, subsample=n_obs_samples)
        ),
    )

    da_subsampled_agg = da_subsampled.reduce(
        func_dict[time_agg], dim="subsample"
    ).median("k")

    for mask in [None, ds.pval_mask]:
        fig, ax = plot_acs_hazard(
            data=da_subsampled_agg,
            stippling=mask,
            title=f"{info.metric} subsampled {time_agg}\n(median of {resamples} samples)",
            date_range=info.date_range,
            cmap=info.cmap,
            cbar_extend="neither",
            ticks=info.ticks,
            tick_labels=None,
            cbar_label=info.units_label,
            dataset_name=f"{info.name} ensemble ({resamples} x max({n_obs_samples} subsample))",
            outfile=f"{info.fig_dir}/{time_agg}_subsampled_{info.filestem(mask)}.png",
            **plot_kwargs,
        )


def plot_obs_anom(
    info,
    ds,
    ds_obs,
    time_agg="maximum",
    metric="anom",
    dparams_ns=None,
    covariate_base=None,
    mask=None,
):
    """Plot map of soft-record metric (e.g., anomaly) between model and obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    metric : {"anom", "anom_std", "anom_pct", "anom_2000yr"}, default "anom"
        Model/obs metric (see `soft_record_metric` for details)
    dparams_ns : xarray.DataArray, optional
        Non-stationary GEV parameters
    covariate_base : int, optional
        Covariate for non-stationary GEV parameters
    mask : xa.DataArray, default None
        Show model similarity stippling mask
    """

    def soft_record_metric(
        info, da, da_obs, time_agg, metric, dparams_ns=None, covariate_base=None
    ):
        """Calculate the difference between two DataArrays."""

        dims = [d for d in da.dims if d not in ["lat", "lon"]]
        da_agg = da.reduce(func_dict[time_agg], dim=dims)
        da_obs_agg = da_obs.reduce(func_dict[time_agg], dim="time")

        # Regrid obs to model grid (after time aggregation)
        da_obs_agg_regrid = general_utils.regrid(da_obs_agg, da_agg)
        anom = da_agg - da_obs_agg_regrid

        kwargs = dict(
            title=f"{time_agg.capitalize()} {info.metric}\ndifference from observed",
            cbar_label=f"Anomaly [{info.units}]",
            cmap=info.cmap_anom,
            ticks=info.ticks_anom,
            cbar_extend="both",
        )

        if metric == "anom_std":
            da_obs_std = da_obs.reduce(np.std, dim="time")
            da_obs_std_regrid = general_utils.regrid(da_obs_std, da_agg)
            anom = anom / da_obs_std_regrid
            kwargs["title"] += " (/σ(obs))"
            kwargs["cbar_label"] = f"Observed\nstandard deviation"

        elif metric == "anom_pct":
            anom = (anom / da_obs_agg_regrid) * 100
            kwargs["cbar_label"] = f"Difference [%]"
            kwargs["title"] += " (%)"
            kwargs["ticks"] = np.arange(-30, 31, 5)

        elif metric == "anom_2000yr":
            covariate = xr.DataArray([covariate_base], dims=info.time_dim)
            rl = eva.get_return_level(2000, dparams_ns, covariate, dims=dims)
            rl = rl.squeeze()
            anom = rl / da_obs_agg_regrid
            kwargs["cbar_label"] = f"Ratio to observed {time_agg}"
            kwargs["title"] = (
                f"Ratio of UNSEEN 2000-year {info.metric}\nto the observed {time_agg}"
            )
            kwargs["ticks"] = np.arange(0.6, 1.45, 0.05)
        return anom, kwargs

    anom, kwargs = soft_record_metric(
        info,
        ds[info.var],
        ds_obs[info.var],
        time_agg,
        metric,
        dparams_ns,
        covariate_base,
    )

    fig, ax = plot_acs_hazard(
        data=anom,
        stippling=mask,
        date_range=info.date_range_obs,
        tick_labels=None,
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/{time_agg}_{metric}_{info.filestem(mask)}.png",
        **kwargs,
        **plot_kwargs,
    )


def plot_event_month_mode(info, ds, mask=None):
    """Plot map of the most common month when event occurs.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model or observational dataset
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    # Calculate month mode
    da = xr.DataArray(
        mode(ds.event_time.dt.month, axis=0).mode,
        coords=dict(lat=ds.lat, lon=ds.lon),
        dims=["lat", "lon"],
    )

    # Map of most common month
    fig, ax = plot_acs_hazard(
        data=da,
        stippling=mask,
        title=f"{info.metric} most common month",
        date_range=info.date_range,
        cmap=plt.cm.gist_rainbow,
        cbar_extend="neither",
        ticks=np.arange(0.5, 12.5),
        tick_labels=list(calendar.month_name)[1:],
        cbar_label="",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/month_mode_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


# def plot_event_month_probability(info, ds, mask=None):
#     """Plot map of the probability of event occurrence in each month.

#     Parameters
#     ----------
#     info : Dataset
#         Dataset information instance
#     ds : xarray.Dataset
#         Model or observational dataset
#     mask : xarray.DataArray, default None
#         Show model similarity stippling mask
#     """
#     # Calculate the probability of event occurrence in each month
#     month_counts = ds.event_time.dt.month.groupby(ds.event_time.dt.month).count(
#         dim=info.time_dim
#     )
#     total_counts = ds.event_time.size
#     month_probabilities = month_counts / total_counts

#     # Map of event occurrence probability
#     fig, ax = plot_acs_hazard(
#         data=month_probabilities,
#         stippling=mask,
#         title=f"{info.metric} event occurrence probability by month",
#         date_range=info.date_range,
#         cmap=plt.cm.gist_rainbow,
#         cbar_extend="neither",
#         ticks=np.arange(0.5, 12.5),
#         tick_labels=list(calendar.month_name)[1:],
#         cbar_label="Probability",
#         dataset_name=info.long_name,
#         outfile=f"{info.fig_dir}/month_probability_{info.filestem(mask)}.png",
#         **plot_kwargs,
#     )


def plot_event_year(info, ds, time_agg="maximum", mask=None):
    """Plot map of the year of the maximum or minimum event.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xarray.Dataset
        Model or observational dataset
    time_agg : {"maximum", "minimum"}, default "maximum"
        Time aggregation function name
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    dt = ds[info.var].copy().compute()
    dt.coords[info.time_dim] = dt.event_time.dt.year

    if time_agg == "maximum":
        da = dt.idxmax(info.time_dim)
    elif time_agg == "minimum":
        da = dt.idxmin(info.time_dim)

    # Map of year of maximum
    fig, ax = plot_acs_hazard(
        data=da,
        stippling=mask,
        title=f"Year of {time_agg} {info.metric}",
        date_range=info.date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="max",
        ticks=np.arange(1960, 2026, 5),  # todo: pass as argument?
        tick_labels=None,
        cbar_label="",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/year_{time_agg}_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def plot_gev_param_trend(info, dparams_ns, param="location", mask=None):
    """Plot map of GEV location and scale parameter trends.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    dparams_ns : xarray.Dataset
        Non-stationary GEV parameters
    param : {"location", "scale"}, default "location"
        GEV parameter to plot
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    var_name = {"location": "loc1", "scale": "scale1"}
    da = dparams_ns.sel(dparams=var_name[param])

    da = da * 10  # Convert to per decade

    fig, ax = plot_acs_hazard(
        data=da,
        stippling=mask,
        title=f"{info.metric} GEV distribution\n{param} parameter trend",
        date_range=info.date_range,
        cmap=info.cmap_anom,
        cbar_extend="both",
        ticks=info.ticks_param_trend[param],
        cbar_label=f"{param.capitalize()} parameter\n[{info.units} / decade]",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/gev_{param}_trend_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def plot_aep(info, dparams_ns, times, aep=1, mask=None):
    """Plot maps of AEP for a given threshold.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    dparams : xarray.Dataset
        Non-stationary GEV parameters
    times : xarray.DataArray
        Start and end years for AEP calculation
    aep : int, default 1
        Annual exceedance probability threshold
    mask : xarray.DataArray, default None
        Show model similarity stippling mask

    Notes
    -----
    * AEP = 1 / RL
    * Plot AEP for times[0], times[1] and the difference between the two.
    """

    ari = eva.aep_to_ari(aep)
    da_aep = eva.get_return_level(ari, dparams_ns, times)

    for i, time in enumerate(times.values):
        fig, ax = plot_acs_hazard(
            data=da_aep.isel({info.time_dim: i}),
            stippling=mask,
            title=f"{info.metric}\n{aep}% Annual Exceedance Probability",
            date_range=time,
            cmap=info.cmap,
            cbar_extend="both",
            ticks=info.ticks,
            tick_labels=None,
            cbar_label=info.units_label,
            dataset_name=info.long_name,
            outfile=f"{info.fig_dir}/aep_{aep:g}pct_{info.filestem(mask)}_{time}.png",
            **plot_kwargs,
        )

    # Time difference (i.e., change in return level)
    da = da_aep.isel({info.time_dim: -1}, drop=True) - da_aep.isel(
        {info.time_dim: 0}, drop=True
    )
    fig, ax = plot_acs_hazard(
        data=da,
        stippling=mask,
        title=f"Change in {info.metric}\n{aep}% Annual Exceedance Probability",
        date_range=f"Difference between {times[0].item()} and {times[1].item()}",
        cmap=info.cmap_anom,
        cbar_extend="both",
        ticks=info.ticks_anom,  # todo: check scale reduced enough
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/aep_{aep:g}pct_{info.filestem(mask)}_{times[0].item()}-{times[1].item()}.png",
        **plot_kwargs,
    )


def plot_aep_empirical(info, ds, aep=1, mask=None):
    """Plot map of empirical AEP for a given threshold.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model or observational dataset
    aep : int, default 1
        Annual exceedance probability threshold
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    ari = eva.aep_to_ari(aep)
    da_aep = eva.get_empirical_return_level(ds[info.var], ari, core_dim=info.time_dim)

    fig, ax = plot_acs_hazard(
        data=da_aep,
        title=f"{info.metric} empirical {aep}%\nannual exceedance probability",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend="both",
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/aep_empirical_{aep:g}pct_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def plot_obs_ari(
    info,
    ds_obs,
    ds,
    dparams_ns,
    covariate_base,
    time_agg="maximum",
    mask=None,
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds_obs : xarray.Dataset
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

    if info.is_model():
        da_obs_agg = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
        da_obs_agg = general_utils.regrid(da_obs_agg, ds[info.var])
        cbar_label = (
            f"Model-estimated\nannual recurrence interval\nin {covariate_base} [years]"
        )
    else:
        da_obs_agg = ds[info.var].reduce(func_dict[time_agg], dim=info.time_dim)
        cbar_label = f"Annual recurrence\ninterval in {covariate_base} [years]"

    rp = xr.apply_ufunc(
        eva.get_return_period,
        da_obs_agg,
        dparams_ns,
        input_core_dims=[[], ["dparams"]],
        output_core_dims=[[]],
        kwargs=dict(covariate=xr.DataArray([covariate_base], dims=info.time_dim)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )

    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    fig, ax = plot_acs_hazard(
        data=rp,
        stippling=mask,
        title=f"Annual recurrence interval\nof observed {info.metric} {time_agg}",
        date_range=info.date_range_obs,
        cmap=cmap,
        cbar_extend="max",
        norm=LogNorm(vmin=1, vmax=10000),
        cbar_label=cbar_label,
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/ari_obs_{time_agg}_{info.filestem(mask)}.png",
        **plot_kwargs,
    )
    return


def plot_obs_ari_empirical(
    info,
    ds_obs,
    ds=None,
    time_agg="maximum",
    mask=None,
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds_obs : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, default None
        Model dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    da_obs_agg = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
    if info.is_model():
        da = ds[info.var]
        da_obs_agg = general_utils.regrid(da_obs_agg, da)
        long_name = f"{info.obs_name}, {info.long_name}"
    else:
        da = ds_obs[info.var]
        long_name = info.obs_name

    rp = eva.get_empirical_return_period(da, da_obs_agg, core_dim=info.time_dim)

    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    fig, ax = plot_acs_hazard(
        data=rp,
        stippling=mask,
        title=f"Empirical annual recurrence interval\nof observed {info.metric} {time_agg}",
        date_range=info.date_range_obs,
        cmap=cmap,
        cbar_extend="max",
        norm=LogNorm(vmin=1, vmax=10000),
        cbar_label="Empirical annual\nrecurrence interval [years]",
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/ari_obs_empirical_{time_agg}_{info.filestem(mask)}.png",
        **plot_kwargs,
    )
    return


def plot_new_record_probability(
    info, ds_obs, ds, dparams_ns, covariate_base, time_agg, ari=10, mask=None
):
    """Plot map of the probability of breaking the obs record in the next X years.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds_obs : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, optional
        Model dataset
    dparams_ns : xarray.DataArray
        Non-stationary GEV parameters
    covariate_base : int
        Covariate for non-stationary GEV parameters (e.g., single year)
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    ari : int, default 10
        Return period in years
    mask : xarray.DataArray, default None
        Show model similarity stippling mask

    Notes
    -----
    * The probability is calculated as 1 - (1 - P(record in a single year))^X
    * The covariate is set to the middle of the year range (covariate + ari/2)
    """

    def new_record_probability(record, dparams_ns, covariate, ari):
        """Probability of exceeding a record in the next {ari} years."""
        shape, loc, scale = eva.unpack_gev_params(dparams_ns, covariate=covariate)
        loc, scale = loc.squeeze(), scale.squeeze()
        # Probability of exceeding the record in a single year
        annual_probability = 1 - genextreme.cdf(record, shape, loc=loc, scale=scale)
        # Probability of exceeding the record at least once over the specified period
        cumulative_probability = 1 - (1 - annual_probability) ** ari
        # Convert to percentage
        probability = cumulative_probability * 100
        return probability

    record = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
    if info.is_model():
        record = general_utils.regrid(record, ds[info.var])

    probability = xr.apply_ufunc(
        new_record_probability,
        record,
        dparams_ns,
        input_core_dims=[[], ["dparams"]],
        output_core_dims=[[]],
        kwargs=dict(
            covariate=xr.DataArray([covariate_base + int(ari / 2)], dims=info.time_dim),
            ari=ari,
        ),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"] * 2,
    )

    fig, ax = plot_acs_hazard(
        data=probability,
        stippling=mask,
        title=f"Probability of record breaking\n{info.metric} in the next {ari} years",
        date_range=f"covariate_base to {covariate_base + ari}",
        baseline=info.date_range,
        cmap=cmap_dict["ipcc_misc_seq_2"],
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label=f"Probability [%]",
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/new_record_probability_{ari}-year_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def plot_new_record_probability_empirical(
    info, ds_obs, ds, time_agg, ari=10, mask=None
):
    """Plot map of the probability of breaking the obs record in the next X years.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds_obs : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, optional
        Model dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    ari : int, default 10
        Return period in years
    mask : xarray.DataArray, default None
        Show model similarity stippling mask

    Notes
    -----
    * empirical based probability - use last 10 years of model data nd % that pass threshold (excluding unsampled final years)
    """

    record = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
    if info.is_model():
        record = general_utils.regrid(record, ds[info.var])

    # delect the latest ari years of data (excluding years that start after last year of init_date )
    max_year = ds.init_date.dt.year.max().load()
    min_year = max_year - ari
    ds_subset = ds.where(
        (ds.time.dt.year.load() >= min_year) & (ds.time.dt.year.load() <= max_year),
        drop=True,
    )
    ds_subset = (ds_subset[info.var] >= record).sum(dim=info.time_dim)
    annual_probability = ds_subset / ds[info.var].time.size * 100
    cumulative_probability = 1 - (1 - annual_probability) ** ari

    # Convert to percentage
    fig, ax = plot_acs_hazard(
        data=cumulative_probability,
        stippling=mask,
        title=f"Probability of record breaking\n{info.metric} in the next {ari} years",
        baseline=info.date_range,
        cmap=plt.cm.BuPu,
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label=f"Probability [%]",
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/new_record_probability_{ari}-year_empirical_{info.filestem(mask)}.png",
        **plot_kwargs,
    )
