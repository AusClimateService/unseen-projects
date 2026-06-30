"""Useful functions."""

import glob
from collections import Counter
import calendar

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xclim as xc

from unseen import array_handling
from unseen import time_utils


def plot_timing(event_starts, model_name):
    """Plot the event timing."""

    if model_name == 'BARRA':
        months = event_starts.dt.month.values
    else:
        event_dates = event_starts.values.flatten()
        months = []
        for date in event_dates:
            if type(date) != float:
                months.append(date.month)

    month_counts = Counter(months)
    months = np.arange(1, 13)
    counts = [month_counts[month] for month in months]

    plt.bar(months, counts)
    plt.title(f'WDDx timing - {model_name}')
    plt.ylabel('number of events')
    plt.xlabel('month')
    xlabels = [calendar.month_abbr[i] for i in months]
    plt.xticks(months, xlabels)
    plt.show()


def ensemble_to_run(model_name, ensnum):
    """Map ensemble number to DCPP model run"""

    ensemble_to_run = {   
        'CanESM5': f'r{ensnum}i1p2f1',
        'CMCC-CM2-SR5': f'r{ensnum}i1p1f1',
        'EC-Earth3': f'r{ensnum}i1p1f1' if ensnum <= 10 else f'r{ensnum-5}i2p1f1',
        'HadGEM3-GC31-MM': f'r{ensnum}i1p1f2',
        'IPSL-CM6A-LR': f'r{ensnum}i1p1f1',
        'MIROC6': f'r{ensnum}i1p1f1',
        'MPI-ESM1-2-HR': f'r{ensnum}i1p1f1',
        'MRI-ESM2-0': f'r{ensnum}i1p1f1',
        'NorCPM1': f'r{ensnum}i1p1f1' if ensnum <= 10 else f'r{ensnum-10}i2p1f1'
    }
    run = ensemble_to_run[model_name]

    return run


def find_dcpp_data(event_df, model_name):
    """Find DCPP data for a dataframe of events."""

    init_adjustment = {
        'CanESM5': -1,
        'CMCC-CM2-SR5': 0,
        'EC-Earth3': 0,
        'HadGEM3-GC31-MM': 0,
        'IPSL-CM6A-LR': -1,
        'MIROC6': 0,
        'MPI-ESM1-2-HR': 0,
        'MRI-ESM2-0': 0,
        'NorCPM1': 0,
    }

    assert model_name in init_adjustment
    
    for index, row in event_df.iterrows():
        init_date = int(row['init_date'].strftime('%Y')) + init_adjustment[model_name]
        ensemble = int(row['ensemble']) + 1
        run = ensemble_to_run(model_name, ensemble)
        start_date = row['event_start'].strftime('%Y-%m-%d')
        wddx_value = row['event_length']
        print(f'{wddx_value} day event starting {start_date}: initialisation year {init_date}, ensemble member {run}')
        available_data = glob.glob(f'/g/data/oi10/replicas/CMIP6/DCPP/*/{model_name}/dcppA-hindcast/s{init_date}-{run}/day/*')
        for path in available_data:
            print(path)


def calc_wddx_timeseries(timeseries, pctl10):
    """Calculate the annual max wind drought duration (WDDx) for a single timeseries"""

    calm_days = timeseries < pctl10
    drought_events = xc.indices.run_length.find_events(calm_days, window=1)
    drought_events = drought_events.assign_coords(
        time=('event', drought_events['event_start'].data)
    ).drop_vars('event_start')
    drought_events = drought_events.swap_dims({'event': 'time'}).dropna('time')
    wddx_da = drought_events['event_length'].resample(time='1YS').max()
    wddx_da = wddx_da.fillna(0.0)
    wddx_times = drought_events['event_length'].resample(time='1YS').map(xr.DataArray.idxmax, dim='time', keep_attrs=True)
    wddx_ds = wddx_da.to_dataset()
    wddx_ds['event_start'] = wddx_times

    return wddx_ds
    
    
def calc_wddx_forecast(lead_indexed_forecast, pctl10):
    """Calculate the WDDx timeseries for a single forecast"""

    time_indexed_forecast = lead_indexed_forecast.swap_dims({'lead_time': 'time'})
    wddx_ds = calc_wddx_timeseries(time_indexed_forecast['sfcWind'], pctl10)
    ntimes = len(wddx_ds['time'])
    wddx_ds['lead_time'] = xr.DataArray(np.arange(1, ntimes + 1, 1), dims={'time': wddx_ds['time']})
    wddx_ds = wddx_ds.swap_dims({'time': 'lead_time'})
    
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
#    wddx_ds = wddx_ds.rename({'time': 'event_start'})

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

