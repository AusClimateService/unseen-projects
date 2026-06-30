"""Process observational and decadal hindcast growing season (Apr-Oct) rainfall datasets.

Defines paths for datafiles and where to save figures.
Contains functions to load and process GSR data from AGCD, CAFE and DCPP models.
Returns datasets of GSR variables `pr`, `decile` or `tercile`.

The function `gsr_data_regions` returns GSR data averaged over the WA and SA regions (stacked along dimension `x`).
The `gsr_data_aus` (calls `gsr_data_aus_AGCD` and `gsr_data_aus_DCPP`) function returna GSR variables on their native grid (subset to Australia south of around 25S).
Can also display information on model grid and sample sizes.
"""

import geopandas as gp
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from unseen.spatial_selection import select_shapefile_regions
from unseen.time_utils import str_to_cftime

# The data and figures subdirs on your system
home = Path("/g/data/xv83/unseen-projects/outputs/wheatbelt")

data_dir = home / "data/"
fig_dir = home / "figures/"

shapefile = home / "shapefiles/australia.shp"
regions = ["WA", "SA"]
data_var = "pr"

# Names of datasets (AGCD, CAFE and DCPP models)
dataset_names = [
    "AGCD",
    "CAFE",
    "CanESM5",
    "CMCC-CM2-SR5",
    "EC-Earth3",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    # "MRI-ESM2-0", # Only 5-year hindcasts available
    "NorCPM1",
]
# Exlude AGCD and list of models
models = [m for m in dataset_names if m not in ["AGCD", "MRI-ESM2-0"]]


def convert_to_quantiles(data, q=3, core_dim="time", quantile_dims="time"):
    """Convert data values to their quantiles.

    Parameters
    ----------
    data : xr.DataArray
        Data to convert to quantiles
    q : int, optional
        Number of quantile bins. Use q=10 for deciles and q=3 for terciles.
    core_dim : str or list of str, optional
        Core dimension to iterate over, by default "time"
    quantile_dims : str or list of str, optional
        Dimensions to calculate quantile bins over, by default "time"

    Returns
    -------
    df : xr.DataArray
        Data converted to quantile category (same shape as input data)

    Notes
    -----
    For ensemble data, determine quantile bins based on all available data  data, core_dim='lead_time' and quantile_dims=["ensemble", "init_date", "lead_time"]
    """

    def cut(ds, bins, **kwargs):
        """Apply pandas.cut, but skip if bins contain duplicates.

        This is useful in case the timeseries, and resulting quantile bins, is empty.
        """
        if np.unique(bins).size < bins.size:
            return ds * np.nan
        return pd.cut(ds, bins=bins, include_lowest=True, **kwargs)

    # Calculate bins that define the range of values in each quantile.
    # The number of bins is q+1 because the bins are the edges of the quantiles.
    bins = data.quantile(q=np.arange(q + 1) / q, dim=quantile_dims)
    bin_labels = np.arange(1, q + 1)

    # Apply the bins to the data to convert to quantiles
    # The output data has the same shape as the input data, but with values replaced by their quantile bin label
    df = xr.apply_ufunc(
        cut,
        data,
        bins,
        input_core_dims=[core_dim, ["quantile"]],
        output_core_dims=[core_dim],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(labels=bin_labels),
        output_dtypes=["float64"],
    )

    # Check if any converted data is missing
    assert np.isnan(data).count() == np.isnan(df).count()
    # Check all quantile labels are present in the output data
    unique_labels = np.unique(df)
    unique_labels = unique_labels[~np.isnan(unique_labels)]
    assert np.all(np.isin(unique_labels, bin_labels)), "Some quantile labels missing"
    # Assert no unexpected quantile labels are present in the output data
    assert np.all(
        np.isin(bin_labels, unique_labels)
    ), "Unexpected quantile labels present"
    return df


def CAFE_datasets_merge_init_months(files):
    """Open and merge CAFE decadal hindcast datasets with different init months.

    Assumes input files are from CAFE model with two initialisations per year
    (May and Nov). This function opens the datasets, adjusts the init_date and
    time coordinates to account for the different initialisation months
    (assumes files made not using --reset_times; times are all YE-DEC;
    DD-12-YYYY => 01-05-YYYY or 01-11-YYYY) and merges them into a dataset.

    Parameters
    ----------
    files : list of str
        List of 2 file paths (May and Nov initialisations).

    Returns
    -------
    ds : xr.Dataset
        Merged dataset with adjusted init_date and time coordinates.
    """

    ds_by_init_month = []

    for t, init_month in enumerate([5, 11]):

        ds = xr.open_dataset(files[t])
        if "lat" in ds.coords:
            # Ensure consistent (recently generated file has lat rounded to 3dp)
            ds.coords["lat"] = ds.lat.round(3)

        # Shift init_date coordinate to correct month (not YE-DEC)
        month_offset = init_month - ds.init_date.dt.month

        assert all(
            month_offset == month_offset[0].item()
        ), "Inconsistent init months in CAFE data"

        month_offset = month_offset[0].item()
        ds["init_date"] = ds.init_date.get_index("init_date").shift(month_offset, "MS")

        # Shift time variable to correct month (not YE-DEC)
        day = 1
        times_new = ds["time"].dt.strftime(f"%Y-{init_month:02d}-{day:02d}")
        # Convert to cftime (original calendar)
        cftime_type = type(ds.time.isel(dict([(k, 0) for k in ds.time.dims])).item())
        times_new = np.vectorize(str_to_cftime)(times_new, cftime_type)
        ds["time"] = (ds["time"].dims, times_new)
        ds["time"].attrs = ds.time.attrs
        ds_by_init_month.append(ds)

    ds = xr.merge(ds_by_init_month, compat="no_conflicts", join="outer")
    return ds


def gsr_data_regions(model, quantile_var="tercile"):
    """Get dataset of Apr-Oct rainfall and quantiles in both regions.

    Parameters
    ----------
    model : str
        Model name (from dataset_names list)
    quantile_var : {"tercile", "decile"}, default "tercile"
        Name of quantile variable to include in dataset

    Returns
    -------
    ds : xr.Dataset
        Dataset containing GSR rainfall and quantile variable.
        Regions are stacked along dimension 'x'.
    """

    if model == "AGCD":
        core_dim, quantile_dims = ["time"], "time"
        model += "-mon"  # Open monthly AGCD data
    else:
        # dim to iterate over when calculating events
        core_dim = ["lead_time"]
        # dims in which to calculate quantiles
        quantile_dims = ["ensemble", "init_date", "lead_time"]

    files = [list(data_dir.glob(f"growing-s*_{model}*{n}.nc")) for n in regions]
    if model == "CAFE":
        # CAFE has two initialisations per year (May and Nov)
        ds_region_list = []
        for i, n in enumerate(regions):
            ds = CAFE_datasets_merge_init_months(files[i]).assign_coords(dict(x=n))
            ds_region_list.append(ds)
    else:

        ds_region_list = [
            xr.open_mfdataset(f).assign_coords(dict(x=n))
            for f, n in zip(files, regions)
        ]

    if "AGCD" not in model:
        # Limit lead times to first 10 for DCPP models (because EC-Earth3 sometimes has 11)
        ds_region_list = [ds.isel(lead_time=slice(0, 10)) for ds in ds_region_list]
        # Drop lead times with all NaNs (NorCPM1 has an extra lead times with all NaNs?)
        ds_region_list = [ds.dropna("lead_time", how="all") for ds in ds_region_list]

    ds = xr.concat(ds_region_list, dim="x")

    for dim in ds[data_var].dims:
        ds[data_var] = ds[data_var].dropna(dim, how="all")

    if quantile_var == "decile":
        q = 10
    elif quantile_var == "tercile":
        q = 3

    ds[quantile_var] = convert_to_quantiles(
        ds[data_var],
        q=q,
        core_dim=core_dim,
        quantile_dims=quantile_dims,
    )
    return ds


def gsr_data_aus(model, quantile_var="tercile"):
    """Get gridded dataset of Apr-Oct rainfall and quantiles in Australia."""

    if model in models:
        ds = gsr_data_aus_DCPP(model, quantile_var=quantile_var)
    elif model == "AGCD":
        ds = gsr_data_aus_AGCD(quantile_var=quantile_var)
    return ds


def gsr_data_aus_AGCD(quantile_var="tercile"):
    """Get AGCD dataset of GSR and quantile data over Australia.

    Parameters
    ----------
    quantile_var : {"tercile", "decile"}, default "tercile
        Name of quantile variable to include in dataset

    Returns
    -------
    ds : xr.Dataset
        Dataset containing GSR rainfall and quantile variable over Australia.
    """

    file_data = data_dir / "growing-season-pr_AGCD-monthly_1900-2022_AMJJASO_gn.nc"

    ds = xr.open_dataset(file_data)[data_var]

    # Apply shapefile mask of Australia
    gdf = gp.read_file(shapefile)
    ds = select_shapefile_regions(ds, gdf)
    # Subset the Australian region south of 25S
    ds = ds.sel(lon=slice(110, 155), lat=slice(-45, -23))
    for dim in ds.dims:
        ds = ds.dropna(dim, how="all")

    ds = ds.to_dataset()

    if quantile_var == "decile":
        q = 10
    elif quantile_var == "tercile":
        q = 3

    ds[quantile_var] = convert_to_quantiles(
        ds[data_var], q=q, core_dim=["time"], quantile_dims="time"
    )
    return ds


def gsr_data_aus_DCPP(model, quantile_var="tercile"):
    """Get DCPP model dataset of GSR, deciles/tercile data over Australia.

    Parameters
    ----------
    model : str
        DCPP model name (from models list)
    quantile_var : {"tercile", "decile"}, default "tercile"
        Name of quantile variable to include in dataset

    Returns
    -------
    ds : xr.Dataset
        Dataset containing GSR rainfall (pr) and quantile variable over Australia.
    """

    assert quantile_var in [
        "decile",
        "tercile",
    ], "quantile must be 'decile' or 'tercile'"
    assert model in models, f"Model {model} not in DCPP models: {models}"

    files = list(data_dir.glob(f"growing-season-pr_{model}*_gn.nc"))

    if model == "CAFE":
        # CAFE has two initialisations per year (May and Nov)
        ds = CAFE_datasets_merge_init_months(files)[data_var]
    else:
        ds = xr.open_dataset(files[0])[data_var]

    # Subset the Australian region (south of 25S)
    ds = ds.sel(lat=slice(-52, -23), lon=slice(105, 155))

    # Remove duplicate latitudes (likely due to rounding errors in gsr output)
    if model in ["CAFE", "EC-Earth3", "NorCPM1"]:
        # Round the data so that all latitudes are the same
        ds.coords["lat"] = ds.lat.round(1)
        # Remove the duplicates. This is a bit of a hack, but it works for
        # this data because it looks like there is only ever two duplicates
        # (one contains the data, the other NaN). So, pick the first duplicate
        # and fill in the NaNs with the second duplicate.
        d0 = ds.drop_duplicates("lat", keep="first")
        d1 = ds.drop_duplicates("lat", keep="last")
        ds = xr.where(np.isnan(d0), d1, d0)

    # Limit lead times to first 10 for DCPP models (because EC-Earth3 sometimes has 11)
    ds = ds.isel(lead_time=slice(0, 10))
    # Drop lead times with all NaNs (NorCPM1 has an extra lead times with all NaNs?)
    ds = ds.dropna("lead_time", how="all")

    # Apply shapefile mask of Australia
    gdf = gp.read_file(shapefile)
    try:
        ds = select_shapefile_regions(ds, gdf, overlap_fraction=0.01)
    except AssertionError:
        # Remove grid spacing check in spatial_selection.fraction_overlap_mask
        # However, it's not necessary as Cartopy will mask the ocean in the plots
        pass

    for dim in ["lat", "lon"]:
        ds = ds.dropna(dim, how="all")

    ds = ds.to_dataset(name=data_var)

    if quantile_var == "decile":
        q = 10
    elif quantile_var == "tercile":
        q = 3

    # Convert data to deciles or terciles
    ds[quantile_var] = convert_to_quantiles(
        ds[data_var],
        q=q,
        core_dim=["lead_time"],
        quantile_dims=["ensemble", "init_date", "lead_time"],
    )
    return ds


def print_model_resolutions():
    """Print spatial resolutions for each model in the regions.

    grid: average distance between grid cells (degrees) at the equator
    nominal res: This is copied from file metadata (dcpp only), 'nominal resolution' is how CMIP6 usually requires models to calculate resolution
    atmos model: This is copied from file metadata (dcpp only), the atmospheric model used and whatever info about the grid that they felt like including
    """

    def get_grid_spacing(ds):
        """Estimate avg grid spacing based on longitude and latitude arrays."""

        # Latitude spacing is constant across the grid
        lat_res = np.mean(np.diff(ds.lat.values))
        # Longitude spacing (in degrees)
        lon_res = np.mean(np.diff(ds.lon.values))
        return lat_res, lon_res

    # Print table header
    print(f"{'Model':15s} {'grid (lat x lon)'} nominal_res atmos_model")

    # Iterate through models: load dataset, print grid spacing and file metadata
    for m in models:
        # Find and open the gridded dataset
        files = list(data_dir.glob(f"growing-season-pr_{m}*_gn.nc"))
        ds = xr.open_dataset(files[0])

        # Calculate avg grid spacing in degrees
        lat_res, lon_res = get_grid_spacing(ds)

        # Check file metadata for nominal resolution and atmos model
        nominal_res = ds.attrs.get("nominal_resolution", "Unknown")

        # Get 'source' attr
        source = ds.attrs.get("source", "Unknown")
        if "atmos" in source:
            # Find string after 'atmos' and before '\n'
            source = source.split("atmos: ")[1].split("\n")[0].strip()

        # Print the output as 'grid:, nominal res, atmos grid'
        print(f"{m:<16s} {lat_res:.3f} x {lon_res:.3f}   {nominal_res:<11} {source}")


def print_sample_sizes():
    """Print sample sizes for each dataset."""

    # Print table header
    print(f"{'Model':13s} #members #leads #inits #samples #(mem x init x leads)")

    for m in dataset_names:
        ds = gsr_data_regions(m)
        if m == "AGCD":
            print(f"{m:36s}{ds.time.size:5d}")
        else:
            n_samples = xr.where(~np.isnan(ds[data_var].isel(x=0)), 1, 0).sum()
            n_samples_est = ds.ensemble.size * ds.init_date.size * ds.lead_time.size

            print(
                f"{m:<16s} {ds.ensemble.size:<6} {ds.lead_time.size:<6} {ds.init_date.size:<6} {n_samples.load().item():<8} {n_samples_est}"
            )
