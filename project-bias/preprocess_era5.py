"""Command line program for ERA5 data pre-processing."""

import glob
import argparse

import numpy as np
import xarray as xr
import xclim
import cmdline_provenance as cmdprov


var_to_cmor_name = {
    'mx2t': 'tasmax',
    'tp': 'pr',
    'msl': 'psl',
    'z': 'z500',
    'latitude': 'lat',
    'longitude': 'lon',
}

cmor_var_attrs = {
    'tasmax': {
        'long_name': 'Daily Maximum Near-Surface Air Temperature',
        'standard_name': 'air_temperature',
    },
    'pr': {
        'long_name': 'Precipitation',
        'standard_name': 'precipitation_flux',
    },
    'psl': {
        'long_name': 'Sea Level Pressure',
        'standard_name': 'air_pressure_at_mean_sea_level',
    },
    'z500': {
        'long_name': '500hPa geopotential height',
        'standard_name': '500hPa_geopotential_height',
    },
    'lat': {
        'long_name': 'latitude',
        'standard_name': 'latitude',
        'axis': 'Y',
        'units': 'degrees_north',
        'bounds': 'lat_bnds'
    },
    'lon': {
        'long_name': 'longitude',
        'standard_name': 'longitude',
        'axis': 'X',
        'units': 'degrees_east',
        'bounds': 'lon_bnds'
    },
}

output_units = {
    'tasmax': 'degC',
    'pr': 'mm d-1',
    'psl': 'hPa',
    'z500': 'm',
}


def convert_units(da, target_units):
    """Convert units.

    Parameters
    ----------
    da : xarray DataArray
        Input array containing a units attribute
    target_units : str
        Units to convert to

    Returns
    -------
    da : xarray DataArray
       Array with converted units
    """

    with xr.set_options(keep_attrs=True):
        da = xclim.units.convert_units_to(da, target_units)
    
    if target_units == 'degC':
        da.attrs['units'] = 'degC'

    return da


def fix_metadata(ds, var):
    "Apply metadata fixes."""

    dims = list(ds[var].dims)
    dims.remove('time')
    units = ds[var].attrs['units']
    for varname in dims + [var]:
        if varname in var_to_cmor_name:
            cmor_var = var_to_cmor_name[varname]
            ds = ds.rename({varname: cmor_var})
        else:
            cmor_var = varname
        ds[cmor_var].attrs = cmor_var_attrs[cmor_var]
    ds[cmor_var].attrs['units'] = units
    ds['time'].attrs['axis'] = 'T'

    return ds


def get_output_encoding(ds, var):
    """Define the output file encoding."""

    encoding = {}
    ds_vars = list(ds.coords) + list(ds.keys())
    for ds_var in ds_vars:
        if ds_var == var:
            encoding[ds_var] = {'_FillValue': np.float32(1e20)}
            encoding[ds_var]['dtype'] = 'float32'
        else:
            encoding[ds_var] = {'_FillValue': None}
            encoding[ds_var]['dtype'] = 'float64'
    encoding['time']['units'] = 'days since 1940-01-01'

    return encoding


def main(args):
    """Run the program."""

    cmor_var = var_to_cmor_name[args.var]
    outdir = '/g/data/xv83/unseen-projects/outputs/bias/data/era5'
    new_log = cmdprov.new_log()
    level = 'pressure-levels' if args.var == 'z' else 'single-levels'    

    for year in np.arange(args.start, 2026):
        infiles = sorted(glob.glob(f'/g/data/rt52/era5/{level}/reanalysis/{args.var}/{year}/*.nc'))
        if not infiles:
            raise OSError(f'No input files for variable {args.var} and year {year}')

        input_ds = xr.open_mfdataset(infiles)
        if args.var == 'mx2t':
            output_ds = input_ds.resample(time='D').max('time', keep_attrs=True)
        elif args.var == 'msl':
            output_ds = input_ds.resample(time='D').mean('time', keep_attrs=True)
        elif args.var == 'z':
            output_ds = input_ds.sel({'level': 500})
            output_ds = output_ds.drop_vars('level')
            output_ds = output_ds.resample(time='D').mean('time', keep_attrs=True)
            output_ds[args.var] = output_ds[args.var] / 9.8
            output_ds[args.var].attrs['units'] = 'm'
        elif args.var == 'tp':
            output_ds = input_ds.resample(time='D').sum('time', keep_attrs=True)
            assert output_ds[args.var].attrs['units'] == 'm'
            output_ds[args.var].attrs['units'] = 'm d-1'
        else:
            raise ValueError(f'Unsupported variable {args.var}')

        output_ds[args.var] = convert_units(output_ds[args.var], output_units[cmor_var])
        output_ds = fix_metadata(output_ds, args.var)
        output_ds = output_ds.isel(lat=slice(None, None, -1))
        output_ds.attrs['history'] = new_log

        outpath = f'{outdir}/{cmor_var}_ERA5_day_gn_{year}0101-{year}1231.nc'
        print(outpath)
        encoding = get_output_encoding(output_ds, cmor_var)
        output_ds.to_netcdf(outpath, encoding=encoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )     
    parser.add_argument("var", type=str, choices=('mx2t', 'tp', 'msl', 'z'), help="input variable")
    parser.add_argument("--start", type=int, default=1940, help="start year [default = 1940]")
    args = parser.parse_args()
    main(args)
