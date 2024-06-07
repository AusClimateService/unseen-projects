"""Open and format AGCD and DCPP growing season (Apr-Oct) rainfall datasets."""

import geopandas as gp
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from unseen.spatial_selection import select_shapefile_regions

# The parent directory data and figures subdirs on your system
home = Path("/g/data/xv83/unseen-projects/outputs/wheatbelt")
# ! Delete before commit
home = Path("/Users/ste785/projects/unseen-projects/outputs/wheatbelt")

# Names of datasets (AGCD and DCPP models)
names = [
    "AGCD",
    "CAFE",
    "CMCC-CM2-SR5",
    "CanESM5",
    "EC-Earth3",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MRI-ESM2-0",
    "NorCPM1",
]
models = [m for m in names if m not in ["AGCD", "MRI-ESM2-0"]]


def apply_cut(data, q=np.arange(11), core_dim="time", bin_dims="time"):
    """Convert data to deciles/terciles.

    Parameters
    ----------
    data : xr.DataArray
        Data to convert to deciles
    q : list of float, optional
        Quantiles to calculate deciles/terciles, by default decile bins
    core_dim : str or list of str, optional
        Core dimension to iterate over, by default "time"
    bin_dims : str or list of str, optional
        Dimensions to calculate decile/tercile bins over, by default "time"

    Returns
    -------
    df : xr.DataArray
        Data converted to decile/terciles (same shape as input data)
    """

    def cut(ds, bins, **kwargs):
        """Apply pandas.cut - skips if bins contain duplicates."""
        if np.unique(bins).size < bins.size:
            return ds * np.nan
        return pd.cut(ds, bins=bins, include_lowest=True, **kwargs)

    bins = data.quantile(q=q / (len(q) - 1), dim=bin_dims)

    df = xr.apply_ufunc(
        cut,
        data,
        bins,
        input_core_dims=[core_dim, ["quantile"]],
        output_core_dims=[core_dim],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(labels=q[1:]),
        output_dtypes=["float64"],
    )

    # Check if any converted data is missing
    assert np.isnan(data).count() == np.isnan(df).count()
    return df


def gsr_data_regions(model, regions):
    """Get Apr-Oct rainfall and decile dataset for WA and SA regions."""
    if model == "AGCD":
        core_dim, bin_dims = ["time"], "time"
        model += "-mon"  # Open monthly AGCD data
    else:
        core_dim = ["lead_time"]
        bin_dims = ["ensemble", "init_date", "lead_time"]

    files = [list(home.glob(f"data/growing-s*_{model}*{n}.nc"))[0] for n in regions]
    ds = xr.concat(
        [xr.open_dataset(f).assign_coords(dict(x=n)) for f, n in zip(files, regions)],
        dim="x",
    )

    for dim in ds.pr.dims:
        ds["pr"] = ds.pr.dropna(dim, how="all")

    ds["decile"] = apply_cut(
        ds.pr,
        q=np.arange(11),
        core_dim=core_dim,
        bin_dims=bin_dims,
    )
    ds["tercile"] = apply_cut(
        ds.pr,
        q=np.arange(4),
        core_dim=core_dim,
        bin_dims=bin_dims,
    )
    return ds


def gsr_data_aus_AGCD():
    """Get Apr-Oct rainfall and decile/tercile dataset for all AGCD grid points."""
    file_data = home / "data/growing-season-pr_AGCD-monthly_1900-2022_AMJJASO_gn.nc"

    ds = xr.open_dataset(file_data).pr

    # Apply shapefile mask of Australia
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    ds = select_shapefile_regions(ds, gdf)
    # Subset the Australian region south of 25S
    ds = ds.sel(lon=slice(110, 155), lat=slice(-45, -23))
    for dim in ds.dims:
        ds = ds.dropna(dim, how="all")

    ds = ds.to_dataset()
    ds["decile"] = apply_cut(ds.pr, q=np.arange(11), core_dim=["time"], bin_dims="time")
    ds["tercile"] = apply_cut(ds.pr, q=np.arange(4), core_dim=["time"], bin_dims="time")
    return ds


def gsr_data_aus_DCPP(model):
    """Get DCPP model dataset of GSR, deciles and tercile data over Australia."""
    file_data = list(home.glob(f"data/growing-season-pr_{model}*_gn.nc"))[0]
    ds = xr.open_dataset(file_data).pr

    # Subset the Australian region (south of 25S)
    ds = ds.sel(lat=slice(-52, -23), lon=slice(105, 155))

    # Remove duplicate latitudes (likely due to rounding errors in gsr output)
    if model in ["EC-Earth3", "NorCPM1"]:
        # Round the data so that all latitudes are the same
        ds.coords["lat"] = ds.lat.round(1)
        # Remove the duplicates. This is a bit of a hack, but it works for
        # this data because it looks like there is only ever two duplicates
        # (one contains the data, the other NaN). So, pick the first duplicate
        # and fill in the NaNs with the second duplicate.
        d0 = ds.drop_duplicates("lat", keep="first")
        d1 = ds.drop_duplicates("lat", keep="last")
        ds = xr.where(np.isnan(d0), d1, d0)

    # Apply shapefile mask of Australia
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    try:
        ds = select_shapefile_regions(ds, gdf, overlap_fraction=0.01)
    except AssertionError:
        # Remove grid spacing check in spatial_selection.fraction_overlap_mask
        # However, it's not necessary as Cartopy will mask the ocean in the plots
        pass

    for dim in ["lat", "lon"]:
        ds = ds.dropna(dim, how="all")

    ds = ds.to_dataset(name="pr")
    # Convert data to deciles and terciles
    ds["decile"] = apply_cut(
        ds.pr,
        q=np.arange(11),
        core_dim=["lead_time"],
        bin_dims=["ensemble", "init_date", "lead_time"],
    )
    ds["tercile"] = apply_cut(
        ds.pr,
        q=np.arange(4),
        core_dim=["lead_time"],
        bin_dims=["ensemble", "init_date", "lead_time"],
    )

    return ds
