# -*- coding: utf-8 -*-
"""UNSEEN hazards configuration."""

import calendar
import geopandas as gp
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr

from unseen import fileio
from acs_plotting_maps import cmap_dict, tick_dict


def get_logger(name="dcpp_file_list_log", level=logging.INFO, home=Path.home()):
    """Initialise a logger that writes to a stream and file."""
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler
    if not logger.handlers:
        handler = logging.FileHandler(f"{home}/{name}.txt")
        handler.setLevel(level)
        fformat = logging.Formatter("{asctime} {message}", "%Y-%m-%d %H:%M", style="{")
        handler.setFormatter(fformat)
        logger.addHandler(handler)
    return logger


datasets = np.array(
    [
        "AGCD",
        "BCC-CSM2-MR",
        "CAFE",
        "CanESM5",
        "CMCC-CM2-SR5",
        "EC-Earth3",
        "HadGEM3-GC31-MM",
        "IPSL-CM6A-LR",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "MRI-ESM2-0",
        "NorCPM1",
    ]
)

func_dict = {
    "mean": np.mean,
    "median": np.nanmedian,
    "maximum": np.nanmax,
    "minimum": np.nanmin,
    "sum": np.sum,
}


hazard_dict = {
    "txx": dict(
        idx="txx",
        metric="TXx",
        var="tasmax",
        var_name="Temperature",
        units="°C",
        units_label="Temperature [°C]",
        # timescale="annual-jul-to-jun",
        freq="YE-JUN",
        obs_name="AGCD",
        cmap=cmap_dict["tasmax"],
        cmap_anom=cmap_dict["anom"],
        ticks=np.arange(22, 58 + 2, 2),
        ticks_anom=np.arange(-5, 5.5, 0.5),
        ticks_param_trend={
            "location": np.arange(-0.06, 0.061, 0.01),
            "scale": np.arange(-0.012, 0.0122, 0.002),
        },
    )
}


class InfoSet:
    """Repository of dataset information to pass to plot functions.

    Parameters
    ----------
    name : str
        Dataset name
    metric : str
        Hazard metric
    file : Path
        Data file path
    obs_file : Path
        Observational data file path
    bias_correction : str, default None
        Bias correction method

    Attributes
    ----------
    name : str
        Dataset name
    metric : str
        Hazard metric
    file : Path
        Forecast file path
    filename : str
        Forecast file name
    obs_file : Path
        Observational file path
    bias_correction : str, default None
        Bias correction method
    fig_dir : Path
        Figure output directory
    hazard : Hazard
        Hazard information
    time_dim : str
        Time dimension name (e.g., "sample" for large ensemble)
    long_name : str
        Dataset long name (i.e., MIROC6 additive bias corrected large ensemble)
    date_range : str
        Date range of dataset (e.g., "01 January 1960 to 31 December 2024")
    date_range_obs : str
        Date range of observational dataset

    """

    def __init__(
        self,
        name,
        metric,
        file,
        obs_file,
        ds=None,
        bias_correction=None,
        masked=None,
        project_dir=Path.home(),
    ):
        """Initialise Dataset instance."""
        self.name = name
        self.metric = metric
        self.file = Path(file)
        self.filestem = self.file.stem
        self.obs_file = Path(obs_file)
        self.bias_correction = bias_correction
        self.fig_dir = f"{project_dir}/figures/{self.metric}"
        self.masked = masked
        if self.masked:
            self.filestem += f"_masked"

        # Get variables from hazard_dict
        for key, value in hazard_dict[metric].items():
            setattr(self, key, value)
        self.cmap_anom.set_bad("lightgrey")
        self.cmap.set_bad("lightgrey")

        # Set dataset-specific attributes
        self.date_range = date_range_str(ds.time, self.freq)
        if self.is_obs():
            self.time_dim = "time"
            self.long_name = f"{self.name}"
        else:
            self.time_dim = "sample"
            self.long_name = f"{self.name} ensemble"
            # self.n_samples = ds[self.var].dropna("sample", how="any")["sample"].size
            if self.bias_correction:
                self.long_name += f" ({self.bias_correction} bias corrected)"
            # else:
            #     self.long_name += f"(samples={self.n_samples})"

    def is_obs(self):
        """Check if dataset is observational."""
        return self.name == self.obs_name

    def __str__(self):
        """Return string representation of Dataset instance."""
        return f"{self.name}"

    def __repr__(self):
        """Return string/dataset representation of Dataset instance."""
        if hasattr(self, "ds"):
            return self.ds.__repr__()
        else:
            return self.name


def get_dataset(
    filename,
    var,
    start_year=None,
    # mask_ocean=False,
    # overlap_fraction=0.1,
    min_lead=None,
    min_lead_kwargs={},
    similarity_file=None,
    alpha=0.05,
):
    """Get metric model dataset.

    Parameters
    ----------
    filename : str
        File path to model data
    var : str
        Variable name
    start_year : int, optional
        Start year
    mask_not_australia : bool, default True
        Apply Australian land-sea mask
    overlap_fraction : float, default 0.1
        Fraction of overlap required for shapefile selection
    min_lead : str or int, optional
        Minimum lead time (file path or integer)
    min_lead_kwargs : dict, optional
        Minimum lead time fileio.open_dataset keyword arguments
    similarity_file : xarray.Dataset, optional
        Similarity mask file path

    alpha : float, default 0.05
        Significance level for AD test

    Returns
    -------
    ds : xarray.Dataset
        Model dataset
    """

    ds = xr.open_dataset(str(filename), use_cftime=True)

    if start_year and "time" in ds[var].dims:
        # Drop years before model data is available
        ds = ds.sel(time=slice(str(start_year), None))

    if "event_time" in ds:
        # Format event time as datetime
        ds["event_time"] = ds.event_time.astype(dtype="datetime64[ns]")

    if min_lead and "lead_time" in ds:
        # Drop lead times less than min_lead
        if isinstance(min_lead, int):
            ds = ds.where(ds["lead_time"] >= min_lead)
        else:
            # Load min_lead from file
            ds_min_lead = fileio.open_dataset(str(min_lead), **min_lead_kwargs)
            min_lead = ds_min_lead["min_lead"].load()
            ds = ds.groupby("init_date.month").where(ds["lead_time"] >= min_lead)
            ds = ds.drop_vars("month")
            ds["min_lead"] = min_lead
        ds = ds.dropna(dim="lead_time", how="all")

    # if mask_ocean:
    #     # Apply Australian land-sea mask
    #     ds = mask_not_australia(ds, overlap_fraction=overlap_fraction)

    if similarity_file:
        # Apply similarity mask
        ds_gof = fileio.open_dataset(str(similarity_file))
        ds["pval_mask"] = ds_gof["ks_pval"] > alpha

    if all([dim in ds.dims for dim in ["init_date", "ensemble", "lead_time"]]):
        # Stack sample dimensions
        ds = ds.stack(sample=["init_date", "ensemble", "lead_time"], create_index=False)
        ds = ds.transpose("sample", ...)
    return ds


# def mask_not_australia(ds, overlap_fraction=0.1):
#     """Apply Australian land-sea mask to dataset."""
#     gdf = gp.read_file(home / "shapefiles/australia.shp")
#     ds = spatial_selection.select_shapefile_regions(
#         ds, gdf, overlap_fraction=overlap_fraction
#     )
#     return ds


def date_range_str(time, freq=None):
    """Return date range 'DD month YYYY' string from time coordinate."""
    # Note that this assumes annual data & time indexed by YEAR_END_MONTH
    if time.ndim > 1:
        # Stack time dimension to get min and max
        time = time.stack(time=time.dims)

    # First and last year
    year = [f(time.dt.year.values) for f in [np.min, np.max]]

    # Index of year end month
    if freq:
        year_end_month = list(calendar.month_abbr).index(freq[-3:].title())
    else:
        year_end_month = time.dt.month[0].item()

    if year_end_month != 12:
        # Times based on end month of year, so previous year is the start
        year[0] -= 1
    YE_ind = [year_end_month + i for i in [1, 0]]
    # Adjust for December (convert 13 to 1)
    YE_ind[1] = 1 if YE_ind[1] == 13 else YE_ind[1]

    # First and last month name
    mon = [list(calendar.month_name)[i] for i in YE_ind]

    day = [1, calendar.monthrange(year[1], YE_ind[1])[-1]]
    date_range = " to ".join([f"{day[i]} {mon[i]} {year[i]}" for i in [0, 1]])
    return date_range


def combine_images(outfile, files):
    fig, axes = plt.subplots(1, 3, figsize=[12, 4], layout="compressed")

    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        img = mpl.image.imread(files[i])
        ax.imshow(img)
        ax.axis(False)
        ax.tick_params(
            axis="both",
            which="both",
            left=False,
            right=False,
            top=False,
            bottom=False,
        )
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=300)
    plt.show()
