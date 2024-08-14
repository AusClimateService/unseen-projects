"""Regrid AGCD-CSIRO Tmax 

https://climate-cms.org/posts/2021-04-09-xesmf-regrid.html
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr
import xesmf as xe


dx = 0.5
home = Path("/g/data/xv83/unseen-projects/outputs/hazards")
out_file = home / "data/tmax_AGCD-CSIRO_r05_1910-2023.nc"

files = "/g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005**.nc"
ds = xr.open_mfdataset(files, use_cftime=True)


ds_out2 = xe.util.grid_2d(
    ds.lon[0].item(), ds.lon[-1].item(), dx, ds.lat[0].item(), ds.lat[-1].item(), dx
)
ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(ds.lat.min(), ds.lat.max() + dx, dx), ds.lat.attrs),
        "lon": (["lon"], np.arange(ds.lon.min(), ds.lon.max(), dx), ds.lon.attrs),
    }
)

regridder = xe.Regridder(ds.tmax, ds_out, "conservative")
dr_out = regridder(ds.tmax, keep_attrs=True)
dr = dr_out.to_dataset()
dr.to_netcdf(out_file)
