# -*- coding: utf-8 -*-
"""Australian Climate Service UNSEEN spatial maps.

Notes
-----
* plot_acs_hazard functions must be modified to allow input colormap
normalisation and plot annotations shifted to compensate multi-line plot titles.
"""

import calendar
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import re
from scipy.stats import genextreme, mode
import xarray as xr

from unseen import eva, general_utils
from acs_plotting_maps import plot_acs_hazard, cmap_dict, tick_dict


plot_kwargs = dict(
    name="ncra_regions",
    mask_not_australia=True,
    figsize=[6.2, 4.6],
    xlim=(113, 153),
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

# Colour blind friendly palette for each month of the year with
# unique, but diverging HSB gradient (cool tones brighter(100, 2) & less
# saturated (-4) than corresponding warm tones)
# https://coolors.co/e0ae6c-cc7349-b82c33-a3147b-721899-002880-1f56ad-3a9bc2-5ad664-99eb8d
month_colours = [
    "#A3147B",  # Jan: Fandango Pink
    "#6F1894",  # Feb: Grape
    "#002880",  # Mar: Resolution Blue
    "#1F56AD",  # Apr: Sapphire
    "#3A9BC2",  # May: Blue Green
    "#5AD664",  # Jun: Malachite
    "#99EB8D",  # Jul: Light Green
    "#F5FFF4",  # Aug: Honeydew
    "#F5E693",  # Sep: Flax
    "#E0AE6C",  # Oct: Earth Yellow
    "#CC7349",  # Nov: Brown Sugar
    "#B82C33",  # Dec: Cardinal Red
]


# month_colours = [
#     "#B53190",  # Jan: Fandango Pink
#     "#842BAB",  # Feb: Grape
#     "#002880",  # Mar: Violet Blue
#     "#3767B3",  # Apr: True Blue
#     "#3A9BC2",  # May: Blue Green
#     "#54D15E",  # Jun: Malachite
#     "#93E087",  # Jul: Light Green
#     "#F5FFF4",  # Aug: Honeydew
#     "#F5E693",  # Sep: Flax
#     "#E8B46F",  # Oct: Earth Yellow
#     "#D97A4E",  # Nov: Burnt Sienna
#     "#C94D53",  # Dec: Bittersweet shimmer
# ]

# https://coolors.co/3a9bc2-3767b3-2f2fa3-74309f-bf359b-f3614a-da8e58-e9b877-59d171-93e087
# https://coolors.co/3a9bc2-3767b3-3c3ccd-74309f-bf359b-df5d5d-e5886c-e9b877-93e087-59d171
month_colours = [
    "#BF359B",  # Jan: Fandango
    "#74309F",  # Feb: Grape
    "#3C3CCD",  # Mar: Palatinate Blue
    "#3767B3",  # Apr: True Blue
    "#3A9BC2",  # May: Blue Green
    "#59D171",  # Jun: Emerald
    "#93E087",  # Jul: Light Green
    "#F5FFF4",  # Aug: Honeydew
    "#F5E693",  # Sep: Flax
    "#E9B877",  # Oct: Earth Yellow
    "#E5886C",  # Nov: Burnt Sienna
    "#DF5D5D",  # Dec: Indian Red
]
month_cmap = mpl.colors.ListedColormap(month_colours)


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
    obs_name : str
        Observational dataset name
    ds : xarray.Dataset, optional
        Model or observational dataset
    obs_ds : xarray.Dataset, optional
        Observational dataset (only if different from ds)
    bias_correction : str, default None
        Bias correction method
    fig_dir : Path
        Figure output directory
    date_dim : str
        Time dimension name for date range (e.g., "sample" or "time")
    kwargs : dict
        Additional metric-specific attributes (idx, var, var_name, units, units_label, freq, cmap, cmap_anom, ticks, ticks_anom, ticks_param_trend, cbar_extend, agcd_mask)

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
        file,
        ds=None,
        obs_name=None,
        obs_ds=None,
        bias_correction=None,
        fig_dir=Path.home(),
        date_dim="time",
        **kwargs,
    ):
        """Initialise class instance."""
        super().__init__()
        self.name = name
        self.file = Path(file)
        self.obs_name = obs_name
        self.bias_correction = bias_correction
        self.fig_dir = Path(fig_dir)

        # Get variables from hazard_dict
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.ds = ds
        self.obs_ds = obs_ds

        # Set dataset-specific attributes
        if ds is not None:
            self.date_range = date_range_str(ds[date_dim], self.freq)
        if obs_ds is not None:
            self.date_range_obs = date_range_str(obs_ds.time, self.freq)
            if ds is None:
                self.date_range = self.date_range_obs
        if self.bias_correction is None:
            self.bc_label = ""
            self.bc_ext = ""
        else:
            self.bc_label = f" ({self.bias_correction} bias corrected)"
            self.bc_ext = f"_{self.obs_name}_{self.bias_correction}_bc"
        if self.is_model():
            self.time_dim = "sample"
            self.long_name = f"{self.name} ensemble"
            self.title_name = self.name
            if self.bias_correction:
                self.long_name += self.bc_label
            # else:
            #     self.n_samples = ds[self.var].dropna("sample", how="any")["sample"].size
            #     self.long_name += f"(samples={self.n_samples})"
            self.long_name_with_obs = f"{self.obs_name}, {self.long_name}"
        else:
            self.time_dim = "time"
            self.long_name = f"{self.name}"
            self.title_name = f"{self.name} gridded observations"
            self.long_name_with_obs = self.long_name

        # Format colour maps
        self.cmap_anom.set_bad("lightgrey")
        self.cmap.set_bad("lightgrey")
        # Set ticks to zero for small values
        self.ticks_anom[np.fabs(self.ticks_anom) < 1e-6] = 0

    def filestem(self, mask=None):
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

    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

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

    da = ds[info.var].reduce(func_dict[time_agg], dim=info.time_dim)

    fig, ax = plot_acs_hazard(
        data=da,
        stippling=mask,
        agcd_mask=info.agcd_mask,
        title=f"{time_agg.capitalize()} {info.metric}",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend=info.cbar_extend,
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/{time_agg}_{info.filestem(mask)}.png",
        savefig=savefig,
        **plot_kwargs,
    )


def resample_subsample(info, ds, time_agg, n_samples, resamples):
    """Return resamples of data aggregated over time."""
    rng = np.random.default_rng(seed=0)

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
        kwargs=dict(size=n_samples, resamples=resamples),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs=dict(output_sizes=dict(k=resamples, subsample=n_samples)),
    )

    da_subsampled_agg = da_subsampled.reduce(
        func_dict[time_agg], dim="subsample"
    ).median("k")
    return da_subsampled_agg


def plot_time_agg_subsampled(info, ds, obs_ds, time_agg="maximum", resamples=1000):
    """Plot map of obs-sized subsample of data (sample median of time-aggregate).

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model dataset
    obs_ds : xarray.Dataset
        Observational dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    resamples : int, default 1000
        Number of random samples of subsampled data
    """

    assert "pval_mask" in ds.data_vars, "Model similarity mask not found in dataset."
    n_obs_samples = obs_ds[info.var].time.size
    da_subsampled_agg = resample_subsample(info, ds, time_agg, n_obs_samples, resamples)

    for mask in [None, ds.pval_mask]:
        fig, ax = plot_acs_hazard(
            data=da_subsampled_agg,
            stippling=mask,
            agcd_mask=info.agcd_mask,
            title=f"{info.metric} {time_agg} in\nobs-sized subsample\n(median of {resamples} resamples)",
            date_range=info.date_range,
            cmap=info.cmap,
            cbar_extend=info.cbar_extend,
            ticks=info.ticks,
            tick_labels=None,
            cbar_label=info.units_label,
            dataset_name=f"{info.long_name} ({resamples} x {time_agg}({n_obs_samples}-year subsample))",
            outfile=f"{info.fig_dir}/{time_agg}_subsampled_{info.filestem(mask)}.png",
            **plot_kwargs,
        )


def soft_record_metric(
    da,
    obs_da,
    time_agg,
    sr_metric,
    plot_dict,
    time_dim="time",
    dparams_ns=None,
    covariate_base=None,
):
    """Calculate the difference between two DataArrays."""

    dims = [d for d in da.dims if d not in ["lat", "lon"]]
    da_agg = da.reduce(func_dict[time_agg], dim=dims)
    obs_da_agg = obs_da.reduce(func_dict[time_agg], dim="time")

    # Regrid obs to model grid (after time aggregation)
    if not all(([da[dim].equals(obs_da[dim]) for dim in ["lat", "lon"]])):
        obs_da_agg_regrid = general_utils.regrid(obs_da_agg, da_agg)
    anom = da_agg - obs_da_agg_regrid

    kwargs = dict(
        title=f"{time_agg.capitalize()} {plot_dict['metric']} difference\nfrom observations",
        cbar_label=plot_dict["units_label"],
        cmap=plot_dict["cmap_anom"],
        ticks=plot_dict["ticks_anom"],
        tick_labels=None,
        cbar_extend="both",
        tick_interval=1,
    )

    if sr_metric == "anom_std":
        obs_da_std = obs_da.reduce(np.std, dim="time")
        obs_da_std_regrid = general_utils.regrid(obs_da_std, da_agg)
        anom = anom / obs_da_std_regrid
        kwargs["title"] += " (standardised)"
        kwargs["cbar_label"] = "Observed\nstandard deviation"
        kwargs["ticks"] = plot_dict["ticks_anom_std"]

    elif sr_metric == "anom_pct":
        anom = (anom / obs_da_agg_regrid) * 100
        kwargs["cbar_label"] = f"{plot_dict['var_name']} difference [%]"
        kwargs["title"] += " (%)"
        kwargs["ticks"] = plot_dict["ticks_anom_pct"]
        kwargs["tick_interval"] = 2

    elif sr_metric == "anom_2000yr":
        covariate = xr.DataArray([covariate_base], dims=time_dim)
        rl = eva.get_return_level(2000, dparams_ns, covariate, dims=dims)
        rl = rl.squeeze()
        anom = rl / obs_da_agg_regrid
        kwargs["cbar_label"] = f"Ratio to observed {time_agg}"
        kwargs["title"] = (
            f"Ratio of 2000-year {plot_dict['metric']}\nto the observed {time_agg}"
        )
        kwargs["ticks"] = plot_dict["ticks_anom_ratio"]
        kwargs["vcentre"] = None
    return anom, kwargs


def plot_obs_anom(
    info,
    ds,
    obs_ds,
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
    obs_ds : xarray.Dataset
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

    plot_dict = dict(
        metric=info.metric,
        var_name=info.var_name,
        units=info.units,
        units_label=info.units_label,
        cmap_anom=info.cmap_anom,
        ticks_anom=info.ticks_anom,
        ticks_anom_std=info.ticks_anom_std,
        ticks_anom_pct=info.ticks_anom_pct,
        ticks_anom_ratio=info.ticks_anom_ratio,
    )

    anom, kwargs = soft_record_metric(
        ds[info.var],
        obs_ds[info.var],
        time_agg,
        metric,
        plot_dict,
        time_dim=info.time_dim,
        dparams_ns=dparams_ns,
        covariate_base=covariate_base,
    )
    fig, ax = plot_acs_hazard(
        data=anom,
        stippling=mask,
        agcd_mask=info.agcd_mask,
        date_range=info.date_range_obs,
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
        agcd_mask=info.agcd_mask,
        title=f"{info.metric} most common month",
        date_range=info.date_range,
        cmap=month_cmap,
        cbar_extend="neither",
        ticks=np.arange(0.5, 13.5),
        tick_labels=list(calendar.month_name)[1:],
        cbar_label="",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/month_mode_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


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
        agcd_mask=info.agcd_mask,
        title=f"Year of {time_agg} {info.metric}",
        date_range=info.date_range,
        cmap=cmap_dict["inferno"],
        cbar_extend="max",
        ticks=np.arange(1960, 2025, 5),
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
        agcd_mask=info.agcd_mask,
        title=f"{info.metric} GEV distribution\n{param} parameter trend",
        date_range=info.date_range,
        cmap=cmap_dict["anom"],
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
            agcd_mask=info.agcd_mask,
            title=f"{info.metric} {aep}% annual\nexceedance probability",
            date_range=time,
            cmap=info.cmap,
            cbar_extend=info.cbar_extend,
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
        agcd_mask=info.agcd_mask,
        title=f"Change in {info.metric} {aep}%\nannual exceedance probability",
        date_range=f"Difference between {times[0].item()} and {times[1].item()}",
        cmap=info.cmap_anom,
        cbar_extend="both",
        ticks=info.ticks_anom,
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
        stippling=mask,
        agcd_mask=info.agcd_mask,
        title=f"{info.metric} empirical {aep}%\nannual exceedance probability",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend=info.cbar_extend,
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/aep_empirical_{aep:g}pct_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def plot_obs_ari(
    info,
    obs_ds,
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

    if ds is not None and not all(
        ([ds[dim].equals(obs_ds[dim]) for dim in ["lat", "lon"]])
    ):
        obs_da_agg = obs_ds[info.var].reduce(func_dict[time_agg], dim="time")
        obs_da_agg = general_utils.regrid(obs_da_agg, ds[info.var])
        cbar_label = (
            f"Model-estimated\nannual recurrence interval\nin {covariate_base} [years]"
        )
    else:
        obs_da_agg = obs_ds[info.var].reduce(func_dict[time_agg], dim=info.time_dim)
        cbar_label = f"Annual recurrence\ninterval in {covariate_base} [years]"

    rp = xr.apply_ufunc(
        eva.get_return_period,
        obs_da_agg,
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
        agcd_mask=info.agcd_mask,
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
    obs_ds,
    ds=None,
    time_agg="maximum",
    mask=None,
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    obs_ds : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, default None
        Model dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    obs_da_agg = obs_ds[info.var].reduce(func_dict[time_agg], dim="time")
    if not all(([ds[dim].equals(obs_ds[dim]) for dim in ["lat", "lon"]])):
        da = ds[info.var]
        obs_da_agg = general_utils.regrid(obs_da_agg, da)
    else:
        da = obs_ds[info.var]

    rp = eva.get_empirical_return_period(da, obs_da_agg, core_dim=info.time_dim)

    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    fig, ax = plot_acs_hazard(
        data=rp,
        stippling=mask,
        agcd_mask=info.agcd_mask,
        title=f"Empirical annual\nrecurrence interval of\nobserved {info.metric} {time_agg}",
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


def nonstationary_new_record_probability(
    record, dparams_ns, start_year, n_years, time_dim="time"
):
    """Calculate the cumulative probability of exceeding a level in a given period."""

    def annual_exceedance_probability(return_level, dparams, covariate):
        """Calculate the annual exceedance probability for a given level and covariate."""

        shape, loc, scale = eva.unpack_gev_params(dparams, covariate=covariate)
        loc, scale = loc.squeeze(), scale.squeeze()
        annual_probability = 1 - genextreme.cdf(
            return_level, shape, loc=loc, scale=scale
        )
        return annual_probability

    def cumulative_aep(return_level, dparams_ns, covariate):
        """Calculate the cumulative probability of exceeding a record in a given period."""

        annual_probabilities = []
        for year in covariate:
            annual_probability = annual_exceedance_probability(
                return_level, dparams_ns, year
            )
            annual_probabilities.append(annual_probability)

        # Combine the annual probabilities to get the cumulative probability
        cumulative_probability = 1 - np.prod(1 - np.array(annual_probabilities))
        return cumulative_probability

    # Create an array of covariate years in the time period
    covariate = xr.DataArray(range(start_year, start_year + n_years + 1), dims=time_dim)

    # Vectorize the cumulative probability function
    cumulative_probability = xr.apply_ufunc(
        cumulative_aep,
        record,
        dparams_ns,
        input_core_dims=[[], ["dparams"]],
        output_core_dims=[[]],
        kwargs=dict(covariate=covariate),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )
    return cumulative_probability


def plot_new_record_probability(
    info, obs_ds, ds, dparams_ns, start_year, time_agg, n_years=10, mask=None
):
    """Plot map of the probability of breaking the obs record in the next X years.

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
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    n_years : int, default 10
        Return period in years
    mask : xarray.DataArray, default None
        Show model similarity stippling mask
    """

    # Get the event record (return period) for the obs data
    record = obs_ds[info.var].reduce(func_dict[time_agg], dim="time")
    if not all(([ds[dim].equals(obs_ds[dim]) for dim in ["lat", "lon"]])):
        record = general_utils.regrid(record, ds)
    cumulative_probability = nonstationary_new_record_probability(
        record, dparams_ns, start_year, n_years, info.time_dim
    )

    baseline = (
        f"{obs_ds.time.dt.year.min().item() - 1} to {obs_ds.time.dt.year.max().item()}"
    )
    fig, ax = plot_acs_hazard(
        data=cumulative_probability * 100,
        stippling=mask,
        agcd_mask=info.agcd_mask,
        title=f"Probability of record breaking\n{info.metric} in the next {n_years} years",
        date_range=f"{start_year} to {start_year + n_years}",
        baseline=baseline,
        cmap=plt.cm.BuPu,
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label="Probability [%]",
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/new_record_probability_{n_years}-year_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def new_record_probability_empirical(
    da, obs_da, n_years, time_agg, time_dim="time", init_dim="init_date"
):
    """Calculate the empirical probability of exceeding a level in a given period."""
    record = obs_da.reduce(func_dict[time_agg], dim="time")

    # Test if the obs and model data is on the same grid
    if not all(([da[dim].equals(obs_da[dim]) for dim in ["lat", "lon"]])):
        record = general_utils.regrid(record, da)

    # Select the latest ari years of data (excluding years that start after last year of init_dim)
    max_year = da[init_dim].dt.year.max().load()
    min_year = max_year - n_years
    da_subset = da.where(
        (da.time.dt.year.load() > min_year) & (da.time.dt.year.load() <= max_year),
        drop=True,
    )
    da_subset = da_subset.dropna(dim=time_dim, how="all")
    da_count = (da_subset > record).sum(dim=time_dim)
    annual_probability = da_count / da_subset[time_dim].size
    cumulative_probability = 1 - (1 - annual_probability) ** n_years
    return da_subset, cumulative_probability


def plot_new_record_probability_empirical(
    info, obs_ds, ds, time_agg, n_years=10, mask=None
):
    """Plot map of the probability of breaking the obs record in the next X years.

    Parameters
    ----------
    info : Dataset
        Dataset information
    obs_ds : xarray.Dataset
        Observational dataset
    ds : xarray.Dataset, optional
        Model dataset
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    n_years : int, default 10
        Return period in years
    mask : xarray.DataArray, default None
        Show model similarity stippling mask

    Notes
    -----
    * empirical based probability - use last 10 years of model data % that pass
    threshold (excluding unsampled final years)
    """

    da_subset, cumulative_probability = new_record_probability_empirical(
        ds[info.var], obs_ds[info.var], n_years, time_agg, time_dim=info.time_dim
    )

    baseline = f"{da_subset.time.dt.year.min().item() - 1} to {da_subset.time.dt.year.max().item()}"
    # Convert to percentage
    fig, ax = plot_acs_hazard(
        data=cumulative_probability * 100,
        stippling=mask,
        agcd_mask=info.agcd_mask,
        title=f"Empirical probability of\nrecord breaking {info.metric}\nin the next {n_years} years",
        baseline=baseline,
        cmap=plt.cm.BuPu,
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label="Probability [%]",
        dataset_name=info.long_name_with_obs,
        outfile=f"{info.fig_dir}/new_record_probability_{n_years}-year_empirical_{info.filestem(mask)}.png",
        **plot_kwargs,
    )


def combine_images(axes, outfile, files, axis=False):
    """Combine plot files into a single figure."""

    for i, ax in enumerate(axes.flatten()):

        if i >= len(files):
            ax.axis("off")
            continue
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

    plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=300)
    plt.show()


def combine_model_plots(metric, bc, obs_name, fig_dir, n_models=12):
    """Combine plots for each model into a single figure.

    Parameters
    ----------
    metric : str
        The metric to combine plots for.
    bias_correction : {None, 'additive', 'multiplicative'}
        The bias correction to combine plots
    obs_name : str, default='AGCD'
        The name of the observations to include in the combined plot.
    n_models : int, default=12
        The number of models to include in each combined plot.
        If there are more than 12 models, the function assumes that the files
        are split by year (i.e., the AEP plots)

    Examples
    --------
     combine_model_plots(
        metric="txx",
        bc="additive",
        obs_name="AGCD",
        fig_dir=f"/g/data/xv83/unseen-projects/outputs/txx/figures/",
        n_models=12,
     )

    Notes
    -----
    * Work in progress
    * Searches for files in the output directory with the following naming convention:
    {prefix}_{metric}_{model}*{bias_correction}{masked}{year}.png
    where, the file name may or may not include the bias correction and year strings.
    * It will ignore un-masked versions of plots if masked versions are present.
    * It will include the observations if they are present in the directory.
    """

    fig_dir = Path(fig_dir)
    files = list(fig_dir.glob(f"*{metric}*.png"))
    files = sorted(files)

    if metric == "txx":
        # Drop any files with "HadGEM3-GC31-MM" (tasmax files have errors)
        files = [f for f in files if "HadGEM3-GC31-MM" not in f.stem]

    # Start of figure file_prefixes (these define separate figures into groups to be combined)
    file_prefixes = np.unique(
        [re.search(f"(.+?)(?=_{metric})", f.stem).group() for f in files]
    )
    file_prefixes = [f for f in file_prefixes if "combined" not in f]

    # Sort filenames into groups that start with the same string
    file_groups = [
        [f for f in files if f.stem.startswith(f"{prefix}_{metric}")]
        for prefix in file_prefixes
    ]

    # Filter by bias correction and masked versions of the figures
    for i, prefix in enumerate(file_prefixes):
        if bc is not None and any([bc in f.stem for f in file_groups[i]]):
            # Keep only original or bias-corrected versions of the figures

            # BC and obs
            file_groups[i] = [
                f
                for f in file_groups[i]
                if (bc in f.stem) or (f"{prefix}_{metric}_{obs_name}" in f.stem)
            ]
        else:
            file_groups[i] = [
                f for f in file_groups[i] if "bias-corrected" not in f.stem
            ]

        # Keep only masked versions of the figures
        if any(["masked" in f.stem for f in file_groups[i]]):
            file_groups[i] = [
                f
                for f in file_groups[i]
                if ("masked" in f.stem) or (f"{prefix}_{metric}_{obs_name}" in f.stem)
            ]
        # Drop any drop_max versions of the figures
        if any(["drop_max" in f.stem for f in file_groups[i]]):
            file_groups[i] = [f for f in file_groups[i] if "drop_max" not in f.stem]

        file_groups[i] = np.array(file_groups[i])
        if "subsampled" in prefix:
            # Add obs max to subsample
            file_groups[i] = [
                list(fig_dir.glob(f"maximum_{metric}_{obs_name}*.png"))[0],
                *file_groups[i],
            ]

        if len(file_groups[i]) == n_models:
            # !!! Shift AGCD to end
            file_groups[i] = [*file_groups[i][1:], file_groups[i][0]]

    # Some files end with numbers eg., _YYYY-YYYY.png or _YYYY.png, so split
    # only these groups further in subgroups based on matching patterns
    file_groups_copy = file_groups.copy()
    file_groups = []
    file_prefixes = []
    for i, group in enumerate(file_groups_copy):
        if len(group) > 0:
            # Check if the group should be split further based on numbers after the last "_"
            group_suffixes = np.unique([f.stem.split("_")[-1] for f in group])
            # Drop any suffixes that dont contain 4 digits in the string
            group_suffixes = [s for s in group_suffixes if re.search(r"\d{4}", s)]

            prefix = np.unique(
                [re.search(f"(.+?)(?=_{metric})", f.stem).group() for f in group]
            )[0]
            if len(group_suffixes) == 0:
                file_groups.append(group)
                file_prefixes.append(prefix)
            else:
                file_subgroups = [
                    [f for f in group if f.stem.endswith(f"_{s}")]
                    for s in group_suffixes
                ]
                for j, subgroup in enumerate(file_subgroups):
                    file_groups.append(subgroup)
                    file_prefixes.append(f"{prefix}_{group_suffixes[j]}")

    # For each file group, combine the images into a single figure
    for i, group in enumerate(file_groups):
        outfile = f"combined_{file_prefixes[i]}_{metric}"
        if bc is not None and any([bc in f.stem for f in group]):
            outfile += f"_{bc}"
        if any(["masked" in f.stem for f in group]):
            outfile += "_masked"
        # Draw axis around these plots
        if any(
            [
                kw in file_prefixes[i]
                for kw in ["similarity", "histograms", "gev_parameters"]
            ]
        ):
            axis = True
        else:
            axis = False

        _, axes = plt.subplots(3, 4, figsize=[12, 10], layout="compressed")
        combine_images(axes, fig_dir / f"acs_combined/{outfile}.png", group, axis=axis)


if __name__ == "__main__":
    # Combine model plots
    metric = "txx"
    for bc in ["additive", None]:
        combine_model_plots(
            metric=metric,
            bc=bc,
            obs_name="AGCD",
            fig_dir=f"/g/data/xv83/unseen-projects/outputs/{metric}/figures/",
            n_models=11,
        )
    metric = "rx1day"
    for bc in ["multiplicative", None]:
        combine_model_plots(
            metric=metric,
            bc=bc,
            obs_name="AGCD",
            fig_dir=f"/g/data/xv83/unseen-projects/outputs/{metric}/figures/",
            n_models=12,
        )
