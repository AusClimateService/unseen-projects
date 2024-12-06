"""Regrid AGCD-CSIRO data."""

import argparse
import datetime
import numpy as np
import xarray as xr
import xesmf as xe


def regrid_dataset(var, files, outfile, dx=0.5, regrid_method="conservative"):
    """Regrid data to a new resolution.

    Parameters
    ----------
    var : str
        Variable name
    files : str
        Data files to regrid (wildcard supported)
    outfile : str
        Regridded data filename (NetCDF)
    dx : float, default 0.5
        Target grid size (degrees)
    regrid_method : {'conservative', 'bilinear', 'nearest_s2d', 'nearest_d2s'}, default 'conservative'
        Regrid method (see xesmf.Regridder)

    Examples
    --------
    regrid_dataset(
        var="tmax",
        files="/g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005**.nc",
        outfile="/g/data/xv83/unseen-projects/outputs/hazards/data/tmax_AGCD-CSIRO_r05_1901-2024.nc",
        dx=0.5,
        regrid_method="conservative",
    )

    python regrid_AGCD-CSIRO.py tmax /g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005**.nc /g/data/xv83/unseen-projects/outputs/hazards/data/tmax_AGCD-CSIRO_r05_1901-2024.nc 0.5 conservative

    """
    ds = xr.open_mfdataset(files, use_cftime=True)

    # Copy attributes
    global_attrs = ds.attrs
    if isinstance(ds, xr.Dataset):
        var_attrs = {v: ds[v].attrs for v in ds.data_vars}

    # Regrid data
    ds_out = xr.Dataset(
        {
            "lat": (
                ["lat"],
                np.arange(ds.lat.min(), ds.lat.max() + dx, dx),
                ds.lat.attrs,
            ),
            "lon": (["lon"], np.arange(ds.lon.min(), ds.lon.max(), dx), ds.lon.attrs),
        }
    )

    regridder = xe.Regridder(ds[var], ds_out, regrid_method)
    ds_regrid = regridder(ds, keep_attrs=True)

    # Update regridded data attributes
    ds_regrid.attrs.update(global_attrs)
    if isinstance(ds_regrid, xr.Dataset):
        for v in ds_regrid.data_vars:
            ds_regrid[v].attrs.update(var_attrs[v])

    # Update history
    ds_regrid.attrs["history"] = (
        f"{datetime.datetime.now().strftime('%a %b %d %H:%M:%S %Y')}: {regrid_method} regrid to {dx}x{dx} degrees, /g/data/xv83/unseen-projects/code/project-hazards/regrid_AGCD-CSIRO.py\n"
        + ds_regrid.attrs["history"]
    )

    # Save regridded data
    ds_regrid.to_netcdf(outfile, compute=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("files", type=str, help="Data files to regrid")
    parser.add_argument("outfile", type=str, help="Data filename")
    parser.add_argument("dx", type=float, default=0.5, help="Target grid (degrees)")
    args = parser.parse_args()

    regrid_dataset(args.var, args.files, args.outfile, dx=args.dx)
