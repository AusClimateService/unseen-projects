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

from unseen import (
    eva,
    general_utils,
    independence,
    similarity,
)
from acs_plotting_maps import plot_acs_hazard, cmap_dict, tick_dict
from cfg import (
    InfoSet,
    get_dataset,
    mask_not_australia,
    func_dict,
    date_range_str,
)


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


def plot_map_time_agg(info, ds, time_agg="maximum"):
    """Plot map of time-aggregated data.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Dataset containing the hazard variable
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    """

    dims = [d for d in ds.dims if d not in ["lat", "lon"]]
    da = ds[info.var].reduce(func_dict[time_agg], dim=dims)

    if info.masked:
        da = da.where(ds.pval_mask)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{time_agg.capitalize()} {info.index}",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend="both",
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/{time_agg}_{info.filestem}.png",
        savefig=False,
        **plot_kwargs,
    )
    plt.savefig(
        f"{info.fig_dir}/{time_agg}_{info.filestem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


def plot_map_time_agg_subsampled(info, ds, ds_obs, time_agg="maximum", n_samples=1000):
    """Plot map of time-aggregated data.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Dataset containing the hazard variable
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Metric to aggregate over
    """

    rng = np.random.default_rng(seed=0)
    n_obs_samples = ds_obs[info.var].time.size

    # Drop model data after last obs year
    da = ds[info.var].where(ds.time.dt.year <= ds_obs.time.dt.year.max(), drop=True)

    if info.masked:
        da = da.where(ds.pval_mask)

    dims = [da[d].size for d in da.dims if d not in [info.time_dim]]
    da_subsampled = np.empty((n_samples, n_obs_samples, *list(dims)))
    for k in range(n_samples):
        for j in range(dims[0]):
            for i in range(dims[1]):
                da_subsampled[k, :, j, i] = rng.choice(
                    da.isel(lat=j, lon=i), n_obs_samples, replace=False
                )

    da_subsampled = xr.DataArray(
        da_subsampled,
        dims=("k", *list(da.dims)),
        coords={
            "k": range(n_samples),
            info.time_dim: ds_obs.time.values,
            "lat": da.lat,
            "lon": da.lon,
        },
    )
    da_subsampled_agg = da_subsampled.reduce(
        func_dict[time_agg], dim=info.time_dim
    ).median("k")

    fig, ax = plot_acs_hazard(
        data=da_subsampled_agg,
        title=f"{info.index} subsampled {time_agg}\n(median of {n_samples} samples)",
        date_range=info.date_range,
        cmap=info.cmap,
        cbar_extend="neither",
        ticks=info.ticks,
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=f"{info.name} ensemble ({n_samples} x max({n_obs_samples} subsample))",
        outfile=f"{info.fig_dir}/{time_agg}_subsampled_{info.filestem}.png",
        **plot_kwargs,
    )


def plot_map_obs_anom(
    info,
    ds,
    ds_obs,
    time_agg="maximum",
    metric="anom",
    dparams_ns=None,
    covariate=None,
):
    """Plot map of soft-record metric (e.g., anomaly) between model and obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xr.Dataset
        Dataset containing the hazard variable
    ds_obs : xr.Dataset
        Dataset containing the observational hazard variable
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    metric : {"anom", "anom_std", "anom_pct", "ratio"}, default "anom"
        Model/obs metric (see `soft_record_metric` for details)
    dparams_ns : xarray.DataArray, optional
        Non-stationary GEV parameters
    covariate : xarray.DataArray, optional
        Covariate for non-stationary GEV parameters

    """

    def soft_record_metric(
        info, da, da_obs, time_agg, metric="anom_std", dparams_ns=None, covariate=None
    ):
        """Calculate the difference between two DataArrays."""

        dims = [d for d in da.dims if d not in ["lat", "lon"]]
        da_agg = da.reduce(func_dict[time_agg], dim=dims)
        da_obs_agg = da_obs.reduce(func_dict[time_agg], dim="time")

        # Regrid obs to model grid (after time aggregation)
        da_obs_agg_regrid = general_utils.regrid(da_obs_agg, da_agg)
        anom = da_agg - da_obs_agg_regrid

        kwargs = dict(
            title=f"{time_agg.capitalize()} {info.index}\ndifference from observed",
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
            covariate = xr.DataArray([covariate], dims=info.time_dim)
            rl = eva.get_return_level(2000, dparams_ns, covariate, dims=dims)
            rl = rl.squeeze()
            anom = rl / da_obs_agg_regrid
            kwargs["cbar_label"] = f"Ratio to observed {time_agg}"
            kwargs["title"] = (
                f"Ratio of UNSEEN 2000-year {info.index}\nto the observed {time_agg}"
            )
            kwargs["ticks"] = np.arange(0.6, 1.45, 0.05)
        return anom, kwargs

    anom, kwargs = soft_record_metric(
        info, ds[info.var], ds_obs[info.var], time_agg, metric, dparams_ns, covariate
    )

    if info.masked:
        anom = anom.where(ds.pval_mask)

    fig, ax = plot_acs_hazard(
        data=anom,
        date_range=info.date_range_obs,
        tick_labels=None,
        dataset_name=f"{info.obs_name}, {info.long_name}",
        outfile=f"{info.fig_dir}/{time_agg}_{metric}_{info.filestem}.png",
        **kwargs,
        **plot_kwargs,
    )


def plot_map_event_month_mode(info, ds):
    """Plot map of the most common month of hazard event.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Dataset containing the hazard variable
    """

    da = xr.DataArray(
        mode(ds.event_time.dt.month, axis=0).mode,
        coords=dict(lat=ds.lat, lon=ds.lon),
        dims=["lat", "lon"],
    )
    if info.masked:
        da = da.where(ds.pval_mask)

    # Map of most common month
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{info.index} most common month",
        date_range=info.date_range,
        cmap=plt.cm.gist_rainbow,
        cbar_extend="neither",
        ticks=np.arange(0.5, 12.5),
        tick_labels=list(calendar.month_name)[1:],
        cbar_label="",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/month_mode_{info.filestem}.png",
        **plot_kwargs,
    )


def plot_map_event_year(info, ds, time_agg="maximum"):
    """Plot map of the year of the maximum or minimum event.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xarray.Dataset
        Dataset containing the hazard variable
    time_agg : {"maximum", "minimum"}, default "maximum"
        Time aggregation function name
    """

    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    dt = ds[info.var].copy().compute()
    dt.coords[info.time_dim] = dt.event_time.dt.year

    if time_agg == "maximum":
        da = dt.idxmax(info.time_dim)
    elif time_agg == "minimum":
        da = dt.idxmin(info.time_dim)
    if info.masked:
        da = da.where(ds.pval_mask)

    # Map of year of maximum
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"Year of {time_agg} {info.index}",
        date_range=info.date_range,
        cmap=cmap_dict,
        cbar_extend="max",
        ticks=np.arange(1960, 2026, 5),  # todo: pass as argument?
        tick_labels=None,
        cbar_label="",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/year_{time_agg}_{info.filestem}.png",
        **plot_kwargs,
    )


def plot_map_gev_param_trend(info, ds, dparams_ns, param="location"):
    """Plot map of GEV location and scale parameter trends.

    Parameters
    ----------
    info : Dataset
        Dataset information instance
    ds : xarray.Dataset
        Model or obs dataset
    dparams_ns : xarray.Dataset
        Non-stationary GEV parameters
    param : {"location", "scale"}, default "location"
        GEV parameter to plot
    """

    var_name = {"location": "loc1", "scale": "scale1"}
    da = dparams_ns.sel(dparams=var_name[param])

    if info.masked:
        da = da.where(ds.pval_mask)

    fig, ax = plot_acs_hazard(
        data=da,
        title=f"{info.index} GEV distribution\n{param} parameter trend",
        date_range=info.date_range,
        cmap=info.cmap_anom,
        cbar_extend="both",
        ticks=info.ticks_param_trend[param],
        cbar_label=f"{param.capitalize()} parameter\n[{info.units} / year]",
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/gev_{param}_trend_{info.filestem}.png",
        **plot_kwargs,
    )


def plot_map_aep(info, ds, dparams_ns, times, aep=1):
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

    Notes
    -----
       - AEP = 1 / RL
       - Plot AEP for times[0], times[1] and the difference between the two.
    """
    ari = eva.aep_to_ari(aep)
    da_aep = eva.get_return_level(ari, dparams_ns, times)
    if info.masked:
        da_aep = da_aep.where(ds.pval_mask)
    for i, time in enumerate(times.values):
        fig, ax = plot_acs_hazard(
            data=da_aep.isel({info.time_dim: i}),
            title=f"{info.index}\n{aep}% Annual Exceedance Probability",
            date_range=time,
            cmap=info.cmap,
            cbar_extend="both",
            ticks=info.ticks,
            tick_labels=None,
            cbar_label=info.units_label,
            dataset_name=info.long_name,
            outfile=f"{info.fig_dir}/aep_{aep:g}pct_{info.filestem}_{time}.png",
            **plot_kwargs,
        )

    # Time difference (i.e., change in return level)
    da = da_aep.isel({info.time_dim: -1}, drop=True) - da_aep.isel(
        {info.time_dim: 0}, drop=True
    )
    fig, ax = plot_acs_hazard(
        data=da,
        title=f"Change in {info.index}\n{aep}% Annual Exceedance Probability",
        date_range=f"Difference between {times[0].item()} and {times[1].item()}",
        cmap=info.cmap_anom,
        cbar_extend="both",
        ticks=info.ticks_anom,  # todo: check scale reduced enough
        tick_labels=None,
        cbar_label=info.units_label,
        dataset_name=info.long_name,
        outfile=f"{info.fig_dir}/aep_{aep:g}pct_{info.filestem}_{times[0].item()}-{times[1].item()}.png",
        **plot_kwargs,
    )


def plot_map_obs_ari(
    info,
    ds,
    ds_obs,
    dparams_ns,
    covariate,
    time_agg="maximum",
):
    """Spatial map of return periods corresponding to the max/min value in obs.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    dparams_ns : xarray.DataArray
        Non-stationary GEV parameters
    covariate : int
        Covariate for non-stationary GEV parameters (single year)
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}, default "maximum"
        Time aggregation function name
    """

    da_agg = ds[info.var].reduce(func_dict[time_agg], dim=info.time_dim)

    if not info.is_obs():
        da_obs_agg = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
        da_obs_agg_regrid = general_utils.regrid(da_obs_agg, da_agg)
        da_agg = da_obs_agg_regrid
        # Mask ocean (for compatibility with dparams_ns)
        da_agg = mask_not_australia(da_agg, overlap_fraction=0.1)
        long_name = f"{info.obs_name}, {info.long_name}"
        cbar_label = (
            f"Model-estimated\nannual recurrence interval\nin {covariate} [years]"
        )
    else:
        long_name = info.obs_name
        cbar_label = f"Annual recurrence\ninterval in {covariate} [years]"

    rp = xr.apply_ufunc(
        eva.get_return_period,
        da_agg,
        dparams_ns,
        input_core_dims=[[], ["dparams"]],
        output_core_dims=[[]],
        kwargs=dict(covariate=xr.DataArray([covariate], dims=info.time_dim)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
    )
    if info.masked:
        rp = rp.where(ds.pval_mask)

    cmap = cmap_dict["inferno"]
    cmap.set_bad("lightgrey")

    fig, ax = plot_acs_hazard(
        data=rp,
        title=f"Annual recurrence interval\nof observed {info.index} {time_agg}",
        date_range=info.date_range_obs,
        cmap=cmap,
        cbar_extend="max",
        norm=LogNorm(vmin=1, vmax=10000),
        cbar_label=cbar_label,
        dataset_name=long_name,
        outfile=f"{info.fig_dir}/ari_obs_{time_agg}_{info.filestem}.png",
        **plot_kwargs,
    )
    return


def plot_map_new_record_probability(
    info, ds, ds_obs, dparams_ns, covariate, time_agg, ari=10
):
    """Plot map of the probability of breaking the obs record in the next X years.

    Parameters
    ----------
    info : Dataset
        Dataset information
    ds : xarray.Dataset
        Model dataset
    ds_obs : xarray.Dataset
        Observational dataset
    dparams_ns : xarray.DataArray
        Non-stationary GEV parameters
    covariate : int
        Covariate for non-stationary GEV parameters (single year)
    time_agg : {"mean", "median", "maximum", "minimum", "sum"}
        Time aggregation function name
    ari : int, default 10
        Return period in years
    """

    # todo: adapt for obs input
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

    da = ds[info.var].reduce(func_dict[time_agg], dim=info.time_dim)
    if info.name != info.obs_name:
        da_obs_agg = ds_obs[info.var].reduce(func_dict[time_agg], dim="time")
        da = general_utils.regrid(da_obs_agg, da)
        # Mask ocean (for compatibility with dparams)
        da = mask_not_australia(da, overlap_fraction=0.1)

    probability = xr.apply_ufunc(
        new_record_probability,
        da,
        dparams_ns,
        input_core_dims=[[], ["dparams"]],
        output_core_dims=[[]],
        kwargs=dict(
            covariate=xr.DataArray([covariate + int(ari / 2)], dims=info.time_dim),
            ari=ari,
        ),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"] * 2,
    )
    if info.masked:
        probability = probability.where(ds.pval_mask)

    fig, ax = plot_acs_hazard(
        data=probability,
        title=f"Probability of\nrecord breaking {info.index}\nin the next {ari} years",
        # date_range=covariate,
        baseline=info.date_range,
        cmap=info.cmap_anom,
        cbar_extend="neither",
        ticks=tick_dict["percent"],
        cbar_label=f"Probability [%]",
        dataset_name=(
            info.obs_name if info.is_obs() else f"{info.obs_name}, {info.long_name}"
        ),
        outfile=f"{info.fig_dir}/new_record_probability_{ari}-year_{info.filestem}.png",
        **plot_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=str, default="txx", help="Hazard index")
    parser.add_argument("dataset", type=str, help="Model name")
    parser.add_argument("--file", type=str, default="AGCD", help="Forecast data file")
    parser.add_argument("--obs_file", type=str, help="Observational data file")
    parser.add_argument("--var", type=str, default="tasmax", help="Variable name")
    parser.add_argument("--start_year", type=int, help="Start year")
    parser.add_argument(
        "--time_agg", type=str, default="maximum", help="Time aggregation method"
    )
    parser.add_argument(
        "--bias_correction", default=None, help="Bias correction method"
    )
    parser.add_argument(
        "--masked", action="store_true", default=False, help="Apply similarity mask"
    )
    parser.add_argument(
        "--overlap_fraction", type=float, default=0.1, help="Overlap fraction"
    )
    parser.add_argument("--similarity_file", type=str, help="Similarity mask file")
    parser.add_argument(
        "--min_lead", default=None, help="Minimum lead time (int or filename)"
    )
    parser.add_argument(
        "--min_lead_kwargs",
        type=str,
        nargs="*",
        default={},
        action=general_utils.store_dict,
        help="Keyword arguments for opening min_lead file",
    )
    args = parser.parse_args()

    # args = argparse.Namespace(
    #     index="txx",
    #     var="tasmax",
    #     dataset=datasets[9],
    #     bias_correction="additive",
    #     masked=False,
    #     overlap_fraction=0.1,
    #     time_agg="maximum",
    #     start_year=1961,  # End of first year
    #     similarity_file=None,
    #     min_lead=None,
    #     min_lead_kwargs={},
    # )

    # Add filenames (todo: define in makefile)
    home = Path("/g/data/xv83/unseen-projects/outputs/hazards")
    obs_file = home / "data/txx_AGCD-CSIRO_r05_1901-2024_annual-jul-to-jun_aus.nc"
    args.file = f"{args.index}_{args.dataset}*_aus.nc"

    if args.dataset != "AGCD" and args.bias_correction is not None:
        args.file = args.file[:-3] + f"*{args.bias_correction}.nc"

    args.file = list(Path(f"{home}/data/").rglob(args.file))[0]
    args.file_s = home / f"data/gev_params_stationary_{args.file.stem}.nc"
    args.file_ns = home / f"data/gev_params_nonstationary_bic_{args.file.stem}.nc"
    if args.dataset != "AGCD":
        # Minimum lead time file
        args.min_lead = list(
            Path(f"{home}/data/").rglob(
                f"independence-test_{args.index}_{args.dataset}*.nc"
            )
        )[0]
        args.min_lead_kwargs = dict(
            shapefile=f"{home}/shapefiles/australia.shp",
            shape_overlap=0.1,
            spatial_agg="median",
        )
        # Similarity file
        args.similarity_file = home / f"data/similarity-test_{args.file.stem}.nc"
        if args.bias_correction is None:
            args.similarity_file = list(
                Path(f"{home}/data/").rglob(
                    f"similarity-test_{args.file.stem}*AGCD-CSIRO_r05.nc"
                )
            )[0]

    if not args.masked:
        args.similarity_file = None

    ds = get_dataset(
        args.file,
        var=args.var,
        start_year=args.start_year,
        min_lead=args.min_lead,
        min_lead_kwargs=args.min_lead_kwargs,
        similarity_file=args.similarity_file,
    )

    # Load info
    info = InfoSet(
        args.dataset,
        args.index,
        args.file,
        obs_file,
        ds=ds,
        bias_correction=args.bias_correction,
        masked=args.masked,
        project_dir=home,
    )

    if args.dataset != "AGCD":
        # Load obs data
        ds_obs = get_dataset(
            obs_file,
            var=args.var,
            start_year=ds.time.dt.year.min().item() - 1,
        )
        info.date_range_obs = date_range_str(ds_obs.time, info.freq)
    else:
        ds_obs = None
        info.date_range_obs = date_range_str(ds.time, info.freq)

    # Load GEV parameters
    covariate = ds["time"].dt.year
    times = xr.DataArray([args.start_year, 2020], dims=info.time_dim)
    year = 2024  # Covariate for non-stationary GEV parameters
    dparams_s = xr.open_dataset(args.file_s)[args.var]
    dparams_ns = xr.open_dataset(args.file_ns)[args.var]

    # Ensure data and dparams are on the same grid (lead-sea mask may crop dparams)
    ds = ds.sel(lat=dparams_ns.lat, lon=dparams_ns.lon)

    # Plot maps
    plot_map_event_month_mode(info, ds)
    plot_map_event_year(info, ds, args.time_agg)
    plot_map_time_agg(info, ds, "median")
    plot_map_time_agg(info, ds, args.time_agg)
    plot_map_gev_param_trend(info, ds, dparams_ns, param="location")
    plot_map_gev_param_trend(info, ds, dparams_ns, param="scale")
    plot_map_aep(info, ds, dparams_ns, times, aep=1)
    plot_map_obs_ari(
        info,
        ds,
        ds_obs,
        dparams_ns,
        covariate=year,
        time_agg=args.time_agg,
    )

    if not info.is_obs():
        # Plot model independence and similarity test maps
        if args.bias_correction is None:
            ds_ind = xr.open_dataset(str(args.min_lead), use_cftime=True)
            independence.spatial_plot(
                ds_ind,
                dataset_name=args.dataset,
                outfile=f"{info.fig_dir}/{args.min_lead.name[:-3]}.png",
            )
        ds_similarity = xr.open_dataset(str(args.similarity_file), use_cftime=True)
        similarity.similarity_spatial_plot(
            ds_similarity,
            dataset_name=info.long_name,
            outfile=f"{info.fig_dir}/{args.similarity_file.name[:-3]}.png",
        )
        # Model-specific plots
        plot_map_time_agg_subsampled(info, ds, ds_obs, args.time_agg, n_samples=1000)
        plot_map_obs_anom(info, ds, ds_obs, "median", "anom")
        plot_map_obs_anom(info, ds, ds_obs, args.time_agg, "anom")
        plot_map_obs_anom(info, ds, ds_obs, args.time_agg, "anom_std")
        plot_map_obs_anom(info, ds, ds_obs, args.time_agg, "anom_pct")
        plot_map_obs_anom(
            info,
            ds,
            ds_obs,
            args.time_agg,
            "anom_2000yr",
            dparams_ns=dparams_ns,
            covariate=year,
        )
        plot_map_new_record_probability(
            info,
            ds,
            ds_obs,
            dparams_ns,
            year,
            args.time_agg,
            ari=10,
        )
