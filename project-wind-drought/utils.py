"""Useful functions."""

import glob
from collections import Counter
import calendar
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xclim as xc

from unseen import array_handling
from unseen import time_utils
from unseen import fileio


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

    norcpm_dict = {
        1: 'r1i1p1f1',
        2: 'r1i2p1f1',
        3: 'r2i1p1f1',
        4: 'r2i2p1f1',
        5: 'r3i1p1f1',
        6: 'r3i2p1f1',
        7: 'r4i1p1f1',
        8: 'r4i2p1f1',
        9: 'r5i1p1f1',
        10: 'r5i2p1f1',
        11: 'r6i1p1f1',
        12: 'r6i2p1f1',
        13: 'r7i1p1f1',
        14: 'r7i2p1f1',
        15: 'r8i1p1f1',
        16: 'r8i2p1f1',
        17: 'r9i1p1f1',
        18: 'r9i2p1f1',
        19: 'r10i1p1f1',
        20: 'r10i2p1f1',
    }

    ecearth_dict = {
        1: 'r1i1p1f1',
        2: 'r2i1p1f1',
        3: 'r3i1p1f1',
        4: 'r4i1p1f1',
        5: 'r5i1p1f1',
        6: 'r6i1p1f1',
        7: 'r6i2p1f1',
        8: 'r7i1p1f1',
        9: 'r7i2p1f1',
        10: 'r8i1p1f1',
        11: 'r8i2p1f1',
        12: 'r9i1p1f1',
        13: 'r9i2p1f1',
        14: 'r10i1p1f1',
        15: 'r10i2p1f1',
    }

    if model_name == 'EC-Earth3':
        run = ecearth_dict[ensnum]
    elif model_name == 'NorCPM1':
        run = norcpm_dict[ensnum]
    else:
        ensemble_to_run = { 
            'BCC-CSM2-MR': f'r{ensnum}i1p1f1',
            'CanESM5': f'r{ensnum}i1p2f1',
            'CMCC-CM2-SR5': f'r{ensnum}i1p1f1',
            'HadGEM3-GC31-MM': f'r{ensnum}i1p1f2',
            'IPSL-CM6A-LR': f'r{ensnum}i1p1f1',
            'MIROC6': f'r{ensnum}i1p1f1',
            'MPI-ESM1-2-HR': f'r{ensnum}i1p1f1',
            'MRI-ESM2-0': f'r{ensnum}i1p1f1',
        }
        run = ensemble_to_run[model_name]

    return run


def check_forecast_match(sfcWind_files, start_str, wddx_value, pctl10, region):
    """Check that the forecast files match the event."""

    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    event_length = int(wddx_value)
    end_date = start_date + timedelta(days=event_length-1)
    end_str = end_date.strftime('%Y-%m-%d')
    ds = fileio.open_dataset(
        sfcWind_files,
        sel={'time': slice(start_str, end_str)},
        shapefile=f'/g/data/xv83/unseen-projects/outputs/wind-drought/shapefiles/{region}.shp',
        shape_overlap=0.1,
        spatial_agg='weighted_mean', 
    )
    assert len(ds['sfcWind'].values) == event_length
    assert np.all(ds['sfcWind'].values < pctl10), 'Incorrect forecast file'


def find_dcpp_data(event_df, model_name, pctl10, region):
    """Find DCPP data for a dataframe of events.

    Parameters
    ----------
    event_df : pandas DataFrame
        List of wind drought events
    model_name : str
        Name of DCPP model
    pctl10: float
        10th percentile of area averaged surface wind speed
    region: {'nem-2030', 'nwis', 'swis', 'se-2030'}
        Name of spatial region

    Returns
    -------
    Paths to forecasts corresponding to the event
    """

    init_adjustment = {
        'BCC-CSM2-MR': 0,
        'CanESM5': -1,
        'CMCC-CM2-SR5': -1,
        'EC-Earth3': -1,
        'HadGEM3-GC31-MM': -1,
        'IPSL-CM6A-LR': -1,
        'MIROC6': -1,
        'MPI-ESM1-2-HR': -1,
        'MRI-ESM2-0': -1,
        'NorCPM1': -1,
    }

    assert model_name in init_adjustment
    
    for index, row in event_df.iterrows():
        init_date = int(row['init_date'].strftime('%Y')) + init_adjustment[model_name]
        ensemble = int(row['ensemble']) + 1
        run = ensemble_to_run(model_name, ensemble)
        start_date = row['event_start'].strftime('%Y-%m-%d')
        wddx_value = row['event_length']
        print(f'{wddx_value} day event starting {start_date}: initialisation year {init_date}, ensemble member {run}')
        sfcWind_files = glob.glob(f'/g/data/oi10/replicas/CMIP6/DCPP/*/{model_name}/dcppA-hindcast/s{init_date}-{run}/day/sfcWind/*/*/*.nc')
        try:
            check_forecast_match(sfcWind_files, start_date, wddx_value, pctl10, region)
        except AssertionError:
            print(f'These appear to be the wrong sfcWind files: {sfcWind_files}')
            continue
        available_data = glob.glob(f'/g/data/oi10/replicas/CMIP6/DCPP/*/{model_name}/dcppA-hindcast/s{init_date}-{run}/*ay/*')
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

