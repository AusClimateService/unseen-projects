"""Useful functions."""

import numpy as np


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
