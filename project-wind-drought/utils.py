"""Useful functions."""

import numpy as np
import xarray as xr
import xclim as xc

from unseen import array_handling
from unseen import time_utils


def calc_wddx_timeseries(timeseries, pctl10):
    """Calculate the annual max wind drought duration (WDDx) for a single timeseries"""

    calm_days = timeseries < pctl10
    drought_events = xc.indices.run_length.find_events(calm_days, window=1)
    drought_events = drought_events.assign_coords(
        time=('event', drought_events['event_start'].data)
    ).drop_vars('event_start')
    drought_events = drought_events.swap_dims({'event': 'time'}).dropna('time')
    wddx_da = drought_events['event_length'].resample(time='1YS').max()
    wddx_times = drought_events['event_length'].resample(time='1YS').map(xr.DataArray.idxmax, dim='time', keep_attrs=True)
    wddx_da['time'] = wddx_times
    wddx_ds = wddx_da.to_dataset()

    return wddx_ds
    
    
def calc_wddx_forecast(lead_indexed_forecast, pctl10):
    """Calculate the WDDx timeseries for a single forecast"""

    time_indexed_forecast = lead_indexed_forecast.swap_dims({'lead_time': 'time'})
    wddx_ds = calc_wddx_timeseries(time_indexed_forecast['sfcWind'], pctl10)
    ntimes = len(wddx_ds['time'])
    wddx_ds['lead_time'] = xr.DataArray(np.arange(1, ntimes + 1, 1), dims={'time': wddx_ds['time']})
    wddx_ds = wddx_ds.swap_dims({'time': 'lead_time'})
    wddx_ds = wddx_ds.reset_coords('time')
    wddx_ds = wddx_ds.rename({'time': 'event_start'})
    
    return wddx_ds


def calc_wddx_model(ds_model, pctl10):
    """Calculate WDDx for a given model"""

    da_sfcWind = array_handling.reindex_forecast(ds_model['sfcWind'])
    time_datetime = np.array(time_utils.cftime_to_str(da_sfcWind.time), dtype='datetime64')
    da_sfcWind = da_sfcWind.assign_coords(time=time_datetime)
    
    ens_list = []
    for ensemble in range(len(ds_model['ensemble'])):
        init_list = []
        for init in range(len(ds_model['init_date'])):
            wddx = calc_wddx_forecast(ds_model.isel({'init_date': init, 'ensemble': ensemble}), pctl10)
            init_list.append(wddx)
        init_concat = xr.concat(init_list, dim='init_date')
        ens_list.append(init_concat)
    ens_concat = xr.concat(ens_list, dim='ensemble')

    return ens_concat


def calc_wddx_obs(ds_obs, pctl10):
    """Calculate WDDX for an observational dataset"""

    wddx_ds = calc_wddx_timeseries(ds_obs['sfcWind'], pctl10)
    wddx_ds = wddx_ds.rename({'time': 'event_start'})

    return wddx_ds


def subset_lat(ds, lat_bnds, lat_dim="lat"):
    """Select grid points that fall within latitude bounds.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
        Input data
    lat_bnds : list
        Latitude bounds: [south bound, north bound]
    lat_dim: str, default 'lat'
        Name of the latitude dimension in ds

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset
    """

    south_bound, north_bound = lat_bnds
    assert -90 <= south_bound <= 90, "Valid latitude range is [-90, 90]"
    assert -90 <= north_bound <= 90, "Valid latitude range is [-90, 90]"

    selection = (ds[lat_dim] <= north_bound) & (ds[lat_dim] >= south_bound)
    ds = ds.where(selection, drop=True)

    return ds


def subset_lon(ds, lon_bnds, lon_dim="lon"):
    """Select grid points that fall within longitude bounds.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
        Input data
    lon_bnds : list
        Longitude bounds: [west bound, east bound]
    lon_dim: str, default 'lon'
        Name of the longitude dimension in ds

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset
    """

    west_bound, east_bound = lon_bnds
    assert west_bound >= ds[lon_dim].values.min()
    assert west_bound <= ds[lon_dim].values.max()
    assert east_bound >= ds[lon_dim].values.min()
    assert east_bound <= ds[lon_dim].values.max()

    if east_bound > west_bound:
        selection = (ds[lon_dim] <= east_bound) & (ds[lon_dim] >= west_bound)
    else:
        selection = (ds[lon_dim] <= east_bound) | (ds[lon_dim] >= west_bound)
    ds = ds.where(selection, drop=True)

    return ds


def model_fixes(ds):
    """Model specific fixes to input data."""

    ds = subset_lat(ds, [-48, -5])
    ds = subset_lon(ds, [105, 160])

    model = ds.attrs['source_id']
    if model == 'CanESM5':
        ds['lat'] = np.round(ds['lat'], 2)
    elif model in ['MPI-ESM1-2-LR', 'EC-Earth3-Veg', 'EC-Earth3']:
        lat_start = np.round(ds.lat.values[0], 2)
        lat_end = np.round(ds.lat.values[-1], 2)
        nlats = len(ds.lat)
        new_lat = np.linspace(lat_start, lat_end, nlats)
        ds['lat'] = new_lat

    return ds

