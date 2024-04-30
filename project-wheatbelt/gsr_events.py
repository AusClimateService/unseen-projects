"""Low/high growing season (Apr-Oct) rainfall event analysis.

- Define events with consecutive years that meet a threshold with no overlapping years 
    (get_events, gsr_events, gsr_events_properties, get_gsr_events_aus)
- Calculate transition probabilities with overlapping years (transition_probability)
- Open data and calculate deciles (get_deciles, get_AGCD_data_regions, get_DCPP_data_regions, get_AGCD_data_aus, get_DCPP_data_aus)
- Apply binomial test to calculate condifence intervals & p-values (binom_ci, apply_binomtest, plot_binom)

Notes
-----
- Calculate (Apr-Oct sum) rainfall for for AGCD monthly data (1900-2022) and DCPP models (1960-)
(global and SA/WA shapefile regions) using the UNSEEN package.
- Calculate deciles, where decile bins are based on all available years 
(excluding model drift)
- Define gsr_events
    - Index 1: Three years in a row at or below 3 decile (make sure the events are not overlapping)
    - Index 2: At least 3 consecutive years under 3 decile
- Transition probability metrics
    - if I had n years of low/high decile, how long to wait until better than average year
    - if I had n years of low/high decile, what is the probability of a better-than-average year next year
    - If I had a year of low/high decile, what is the probability the year next year will be the same (persistence probability)
- Pot timeseries of deciles with events shaded
 Plot frequency of events
- Plot median and maxium duration of consecutive years
- Plot median and maxium of Apr-Oct rainfall during events
"""

from dataclasses import dataclass
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
import xarray as xr

from unseen.spatial_selection import select_shapefile_regions

home = Path("/g/data/xv83/unseen-projects/outputs/wheatbelt")

# Names for DCPP models (and AGCD)
models = [
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


@dataclass
class Events:
    """Class for keeping track of event details."""

    min_duration: int
    fixed_duration: bool
    minimize: bool
    operator: str

    def __init__(
        self,
        min_duration: int,
        operator: str,
        fixed_duration: bool = True,
        minimize: bool = False,
    ):
        self.min_duration = min_duration
        self.operator = operator
        self.fixed_duration = fixed_duration
        self.minimize = minimize

        # For plot/file labels
        if self.operator == "less":
            self.event_type = "LGSR"
            self.threshold = 3
            self.sym = "≤"
        else:
            self.event_type = "HGSR"
            self.threshold = 8
            self.sym = "≥"

        if not self.fixed_duration:
            self.event_type += "_max"

        if not self.minimize and self.fixed_duration:
            self.event_type += "_first"

        self.decile = f"{self.sym}{self.threshold} decile"
        self.name = f"{self.min_duration}yr {self.decile}"


def gsr_events(data, decile, time_dim="time", **kwargs):
    """Get events & event properties (vectorize get_events & get_event_properties)."""

    events = xr.apply_ufunc(
        get_events,
        decile,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=["float64"],
    )
    variables = [
        "id",
        "index_start",
        "index_end",
        "duration",
        "time_start",
        "time_end",
        "gsr_mean",
        "gsr_max",
        "gsr_min",
        "grs_next",
        "decile_next",
    ]

    # Max number of events (for length of dimension)
    n_events = events.max().load().item()

    # Convert times to numeric (days since epoch - avoids apply_ufunc dtype error)
    if data["time"].dtype == "datetime64[ns]":
        epoch = np.datetime64("1900-01-01T00:00:00")
        time = (data["time"] - epoch) / np.timedelta64(1, "D")
    else:
        # Assumes "time" is the actual time variable
        time = data["time"]
    dtypes = []
    for v in variables:
        dtype = "float64" if "time" not in v else str(time.dtype)
        dtypes.append(dtype)

    da_list = xr.apply_ufunc(
        get_event_properties,
        data.chunk({time_dim: -1}),
        decile.chunk({time_dim: -1}),
        events,
        time.chunk({time_dim: -1}),
        input_core_dims=[[time_dim], [time_dim], [time_dim], [time_dim]],
        output_core_dims=[["event"] for _ in range(len(variables))],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(n_events=n_events, variables=variables),
        output_dtypes=dtypes,  # ["float64"] * len(variables),
        dask_gufunc_kwargs={"output_sizes": {"event": n_events}},
    )

    # Create dataset from output DataArrays
    ds = xr.Dataset(coords={"event": np.arange(n_events)})
    for v, da in zip(variables, da_list):
        if "time" in v and data["time"].dtype == "datetime64[ns]":
            # Convert times back to datetime64[ns]
            ds[v] = da * np.timedelta64(1, "D") + epoch
        else:
            ds[v] = da
    return events, ds


def get_event_properties(
    data,
    decile,
    events,
    time,
    n_events,
    variables,
):
    """Get consecutive year event properties (use with apply_func for extra dims).

    Parameters
    ----------
    data : array-like
        Rainfall timeseries
    decile : array-like
        Decile timeseries
    events : xa.DataArray
        Time series of labeled events
    dim : str, optional
        Time dimension name, by default "time"

    Returns
    -------
    list of xarray.Dataset
        DataArrays of event properties (ragged arrays)
    """
    # Create dataset to store event properties
    ds = xr.Dataset(coords={"event": np.arange(n_events)})

    for v in variables:
        if "time" in v:
            dtype = str(time.dtype)  # "datetime64[ns]"
        else:
            dtype = "float64"
        ds[v] = xr.DataArray(
            np.full((ds.event.size), np.nan, dtype=dtype), dims=["event"]
        )

    # Dict of event indexes (includes 0, which is not an event)
    val_inds = scipy.ndimage.value_indices(events.astype(dtype=int))
    # Loop through each event and calculate properties
    for ev in list(val_inds.keys())[:-1]:

        inds = val_inds[ev + 1][0]
        dx_ev = data[inds]

        loc = {"event": ev}
        ds["id"][loc] = ev
        ds["index_start"][loc] = inds[0]
        ds["index_end"][loc] = inds[-1]

        ds["duration"][loc] = len(inds)
        ds["time_start"][loc] = time[inds][0]
        ds["time_end"][loc] = time[inds][-1]

        ds["gsr_mean"][loc] = np.mean(dx_ev)
        ds["gsr_max"][loc] = np.max(dx_ev)
        ds["gsr_min"][loc] = np.min(dx_ev)

        if inds[-1] + 1 < len(data):
            ds["grs_next"][loc] = data[inds[-1] + 1]  # Next year pr
            ds["decile_next"][loc] = decile[inds[-1] + 1]  # Next year decile

    return tuple([ds[v] for v in variables])


def get_events(
    d,
    threshold,
    min_duration,
    operator="less",
    fixed_duration=True,
    minimize=True,
):
    """Label contiguous regions of d <=/=> decile_threshold with duration >= min_duration.

    Parameters
    ----------
    d : xr.DataArray
        1D array of decile values
    decile_threshold : int
        Decile threshold for an event
    min_duration : int
        Minimum duration of event.
    operator : {"less", "greater"}, optional
        Operator to apply a threshold
    fixed_duration : bool, optional
        Define events that are always min_duration long
    minimize : bool, optional
        Select events within contiguous regions with the lowest/highest overall value

    Notes
    -----
    - Calculates events in which there are a minimum number of values in
    a row at or below/above a threshold (no overlapping years)
    - When there are more than the minimum consecutive values, the events
    are chosen to minimize/maximize their overall value, whilst retaining
    the maximum number of events
    - Used for plots of frequency, avg rainfall & max duration of events

    Example
    -------
    - Find events of duration >= min_duration:
        get_events(d, t, min_duration, fixed_duration=False, minimize=True)
    - Find events of duration == min_duration:
        get_events(d, t, min_duration, fixed_duration=True, minimize=True)
    """

    if np.isnan(d).all():
        # Return NaNs if all input data is NaN
        return d * np.nan

    # Threshold mask
    if operator == "less":
        m = d <= threshold
    else:
        m = d >= threshold

    # Find contiguous regions of m = True
    events_init, n_events = scipy.ndimage.label(m)
    events = events_init.copy()

    # Find all events of duration >= thresholdin_duration
    for ev in range(1, n_events + 1):
        ev_mask = events_init == ev

        duration = (ev_mask).sum()
        inds = np.arange(len(d), dtype=int)[ev_mask]

        # Drop labels with duration < min_duration & reset count
        if duration < min_duration:
            # Delete event number & update event numbers for the rest of the array
            events[inds[0] :] -= 1
            events[inds] = 0
            events[events_init == 0] = 0
            continue

        if fixed_duration:
            # Maximum number of min_duration chunks
            max_sub_events = duration // min_duration
            remainder = duration % min_duration

            # Event meets criteria (no change) or should be evenly split into min_duration chunks
            if remainder == 0 or not minimize:
                for i in range(max_sub_events):
                    # Increase id of following events
                    index_next = inds[0] + i * min_duration
                    events[index_next:] += min(i, 1)
                    events[events_init == 0] = 0

                if remainder > 0:
                    # Set value of remaining timesteps to zero
                    events[inds[-remainder] : inds[-remainder] + remainder] = 0

            # Pick the lowest decile events within regions of consecutive values
            else:
                d_subset = d[ev_mask]
                # Indexes of the start of a 3-year run in the subset
                inds_subset = np.arange(duration - min_duration + 1, dtype=int)
                # 3-year sum starting at each index in the subset
                ev_sum = np.array(
                    [sum(d_subset[i : i + min_duration]).item() for i in inds_subset]
                )

                # Find the lowest 3-year sum in the subset (duration less than 2x min_duration)
                if max_sub_events == 1:
                    if operator == "less":
                        i = np.argmin(ev_sum)  # Index of lowest 3-year sum
                    else:
                        i = np.argmax(ev_sum)  # Index of lowest 3-year sum
                    events[
                        inds[
                            (i > np.arange(duration))
                            | (np.arange(duration) >= i + min_duration)
                        ]
                    ] = 0

                # Find the max number of min_duration chunks with the lowest overall value.
                # i.e., 10 consecutive times will be split into 3 non-overlapping events with the minimum overall total.
                else:
                    # Finds the minimum 3-year sum, remove these elements and repeat to find the next smallest 3-year sum.
                    # Start at the jth lowest sum to find different combinations (i.e., because
                    # selecting the smallest 3-year sum may not be the smallest overall option).
                    ev_alts = []
                    for j in range(duration - min_duration + 1):
                        # Reset arrays for each combination trial.
                        ev_sum_ = ev_sum.copy()
                        inds_subset_ = inds_subset.copy()
                        ev_inds = []  # event start index

                        # Find indexes for each 3-year chunk in the subset
                        for k in range(max_sub_events):
                            sort_inds = np.argsort(ev_sum_)
                            if operator == "greater":
                                sort_inds = sort_inds[::-1]

                            if k == 0:
                                sort_inds = sort_inds[j:]  # Start at jth lowest chunk
                            i = inds_subset_[sort_inds[0]]

                            # Drop the indexes of this event (and the previous 2 values in ev_sum, as
                            # they are based on days that would overlap with the current event)
                            inds_to_drop_mask = (
                                inds_subset_ < max(i - (min_duration - 1), 0)
                            ) | (inds_subset_ > i + (min_duration - 1))
                            ev_sum_ = ev_sum_[inds_to_drop_mask]
                            inds_subset_ = inds_subset_[inds_to_drop_mask]
                            ev_inds.append(i)

                            # Stop if there are not enough days left to make another event
                            if len(inds_subset_) == 0:
                                break

                        # Add the events to the list of options if there are enough events
                        if len(ev_inds) >= max_sub_events:
                            ev_alts.append(ev_inds)

                    # Drop combinations that don't have unique event start & end indexes (probably can delete this line)
                    ev_alts = np.array(ev_alts)

                    ev_alts = ev_alts[
                        [
                            np.unique([v[i] for i in range(len(v))]).size
                            == max_sub_events
                            for v in ev_alts
                        ]
                    ]

                    try:
                        # Select the minimum sum of the lowest 3-year sums in the subset
                        alts = [
                            np.sum([ev_sum[v[i]] for i in range(max_sub_events)])
                            for v in ev_alts
                        ]
                        if operator == "less":
                            ev_inds = ev_alts[np.argmin(alts)]
                        else:
                            ev_inds = ev_alts[np.argmax(alts)]

                    except Exception as e:
                        print(ev_inds, alts, max_sub_events)
                        print(e)

                    # Reset & assign an event number for each event found in the subset
                    # Current event number (ev doesn't get updated)
                    ev_id = events[ev_mask][0]
                    new = events[ev_mask].copy() * 0  # Set all events in subset to 0
                    for a, i in enumerate(sorted(ev_inds)):
                        new[i : i + min_duration] = ev_id + a  # Assign new event number
                    events[ev_mask] = new
                    # Increase event numbers for the rest of the array
                    events[inds[-1] + 1 :] += a

        #  Reset non-event values to 0
        events[events_init == 0] = 0
    return events


def get_deciles(data, core_dim="time", decile_dims="time"):
    """Convert data to deciles.

    Parameters
    ----------
    data : xr.DataArray
        Data to convert to deciles
    core_dim : str or list of str, optional
        Core dimensions, by default "time"
    decile_dims : str or list of str, optional
        Dimensions to calculate decile bins over, by default "time"

    Returns
    -------
    deciles : xr.DataArray
        Decile values (same shape as input data)

    Examples
    --------
    # Calculate deciles for an ensemble, with bins based on all data at a point
    decile = get_ensemble_deciles(
        data,
        core_dim=["lead_time"],
        decile_dims=["init_date", "ensemble", "lead_time"]
    )
    """

    def cut(ds, bins, **kwargs):
        """Apply pandas.cut - skips if bins contain dupilicates."""
        if np.unique(bins).size < bins.size:
            return ds * np.nan
        return pd.cut(ds, bins=bins, include_lowest=True, **kwargs)

    q = np.arange(11)
    decile_bins = data.quantile(q=q / 10, dim=decile_dims)

    decile = xr.apply_ufunc(
        cut,
        data,
        decile_bins,
        input_core_dims=[core_dim, ["quantile"]],
        output_core_dims=[core_dim],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(labels=q[1:]),
        output_dtypes=["float64"],
    )

    assert np.isnan(data).count() == np.isnan(decile).count()
    return decile


def get_AGCD_data_regions(regions):
    """Get Apr-Oct rainfall and decile dataset for WA and SA regions."""
    files = [list(home.glob(f"data/growing-s*_AGCD-mon*{n}.nc"))[0] for n in regions]
    data = xr.concat(
        [xr.open_dataset(f).assign_coords(dict(x=n)) for f, n in zip(files, regions)],
        dim="x",
    ).pr
    decile = get_deciles(data, core_dim=["time"], decile_dims="time")
    return data, decile


def get_DCPP_data_regions(model, regions):
    """Get Apr-Oct rainfall and decile dataset for WA and SA regions."""
    files = [list(home.glob(f"data/growing-s*_{model}*{n}.nc"))[0] for n in regions]
    data = xr.concat(
        [xr.open_dataset(f).assign_coords(dict(x=n)) for f, n in zip(files, regions)],
        dim="x",
    ).pr
    decile = get_deciles(
        data, core_dim=["lead_time"], decile_dims=["init_date", "ensemble", "lead_time"]
    )
    return data, decile


def get_AGCD_data_au():
    """Get Apr-Oct rainfall and decile dataset for all AGCD grid points."""
    file_data = home / "data/growing-season-pr_AGCD-monthly_1900-2022_AMJJASO_gn.nc"

    data = xr.open_dataset(file_data).pr

    # Apply shapefile mask of Australia
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    data = select_shapefile_regions(data, gdf)

    decile = get_deciles(data, core_dim=["time"], decile_dims="time")

    data = data.where(decile.notnull())
    data = data.sel(lon=slice(110, 155))

    for dim in data.dims:
        data = data.dropna(dim, how="all")
        decile = decile.dropna(dim, how="all")

    return data, decile


def get_DCPP_data_au(model):
    """Get GSR and decile data for a DCPP model over Australia."""
    file_data = list(home.glob(f"data/growing-season-pr_{model}*_gn.nc"))[0]
    data = xr.open_dataset(file_data).pr
    # Apply shapefile mask of Australia
    data = data.sel(lat=slice(-50, -10), lon=slice(105, 155))
    gdf = gp.read_file(home / "shapefiles/australia.shp")
    try:
        data = select_shapefile_regions(data, gdf, overlap_fraction=0.01)
    except AssertionError:
        data = select_shapefile_regions(data, gdf)
    for dim in ["lat", "lon"]:
        data = data.dropna(dim, how="all")

    decile = get_deciles(
        data, core_dim=["lead_time"], decile_dims=["init_date", "ensemble", "lead_time"]
    )

    return data, decile


def get_events_au(data, decile, event, model, time_dim="time"):
    """Get GSR event property dataset for all grid points."""

    def convert_time(time_start, time_end):
        """Convert time_start and time_end to datetime64[ns]."""
        if not isinstance(time_start, (float, int)) and pd.notnull(time_start):
            time_start = np.datetime64(time_start.isoformat())
            time_end = np.datetime64(time_end.isoformat())
        else:
            time_end = pd.NaT
            time_start = pd.NaT
        return time_start, time_end

    file_events = (
        home / f"data/{event.event_type}_{event.min_duration}yr_events_aus_{model}.nc"
    )

    if file_events.exists():
        ds = xr.open_dataset(file_events)

    else:
        _, ds = gsr_events(
            data,
            decile,
            time_dim=time_dim,
            threshold=event.threshold,
            min_duration=event.min_duration,
            fixed_duration=event.fixed_duration,
            minimize=event.minimize,
            operator=event.operator,
        )

        if ds["time_start"].dtype == "object":
            # Convert time_start and time_end to datetime64[ns]
            ds["time_start"], ds["time_end"] = xr.apply_ufunc(
                convert_time,
                ds.time_start,
                ds.time_end,
                input_core_dims=[[], []],
                output_core_dims=[[], []],
                vectorize=True,
                dask="parallelized",
            )
        ds = ds.load()
        ds.to_netcdf(file_events)

    # Check event durations meets criteria
    duration = ds.duration.values
    if event.fixed_duration:
        assert np.all(duration == event.min_duration, where=~np.isnan(duration))
    else:
        assert np.all(duration >= event.min_duration, where=~np.isnan(duration))

    return ds


def event_inds(m, min_duration):
    """Get indexes of min_duration events in masked decile timeseries."""
    if isinstance(m, xr.DataArray):
        m = m.values  # Only works for numpy arrays

    if min_duration == 1:
        inds = np.flatnonzero(m)
    elif min_duration == 2:
        inds = np.flatnonzero(m[:-1] & m[1:])
        # Drop consecutive "False" events
        inds = inds[m[inds]]
    elif min_duration == 3:
        inds = np.flatnonzero(m[:-2] & m[1:-1] & m[2:])
        # Drop consecutive "False" events
        inds = inds[m[inds]]
    return inds


def transition_probability(
    decile, threshold, operator, min_duration, time_dim="time", binned=True
):
    """Calculate the probability of transitioning to another year of a low/high decile.

    Parameters
    ----------
    decile : xa.DataArray
        Decile values
    threshold : float
        Decile threshold
    operator : {"less", "greater"}
        Operator to apply a threshold
    min_duration : int
        Minimum duration of event
    time_dim : str, optional
        Name of the time dimension, by default "time"
    binned : bool, optional
        Bin the deciles into dry/medium/wet or 1-10, by default True

    Returns
    -------
    k : float or xr.DataArray
        Number of next year deciles (q=dry, medium, wet)
    n : float or xr.DataArray
        Total number of next year deciles
    bins : array-like
        Bin edges of output

    Notes
    -----
    - Calculates the probability of transitioning from n years in a row
    above/below a decile threshold to other deciles (i.e., dry, medium, wet).
    - Note that this includes overlapping years (unlike gsr_events).
    - If the last year in the series meets the criteria, it is dropped from
    the total event count next year because the 'next year' is not known.
    - Used for persistance_probability, transistion_probability and
    transtion_matrix plots.
    """
    assert min_duration <= 3

    def bin_decile_next(m, decile, min_duration, bins):
        """Find indexes of min_duration events and bin the next year deciles."""
        inds = event_inds(m, min_duration)

        # Indexes of following years that meet criteria
        inds_next = np.array(list(inds)) + min_duration
        # Drop indexes that are out of bounds
        inds_next = inds_next[inds_next < len(m)]

        # Return zeros if there are no events
        if inds_next.size == 0:
            k = np.zeros(len(bins) - 1, dtype=int)
            return k, 0

        # Get the deciles of the following years
        decile_next = decile[inds_next]

        # Total number of next year deciles
        total = decile_next.size

        # Bin the next year deciles
        k, _ = np.histogram(decile_next, bins=bins)
        return k, total

    # Create decile threshold mask
    if operator == "less":
        m = decile <= threshold
    else:
        m = decile >= threshold

    # Create bins for deciles (dry/medium/wet) or bin each decile
    bins = np.array([1, 4, 8, 11]) if binned else np.arange(1, 12)

    k, n = xr.apply_ufunc(
        bin_decile_next,
        m,
        decile,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[["q"], []],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(min_duration=min_duration, bins=bins),
        output_dtypes=["int"] * len(bins),
    )

    return k, n, bins


def transition_time(decile, min_duration, time_dim="time", transition_from="dry"):
    """Count the transition times between n-year dry/wet events and next high/low decile year.

    Parameters
    ----------
    decile : xa.DataArray
        Decile values
    min_duration : int
        Minimum duration of event
    time_dim : str, optional
        Name of the time dimension, by default "time"
    transition_from : {"dry", "wet"}, optional
        Transition from dry or wet events, by default "dry"

    Returns
    -------
    k : float or xr.DataArray
        Count of years between low/high events
    bins : array-like
        Bin edges of output
    """
    assert min_duration <= 3

    def transition_years(m0, m1, min_duration, bins):
        """Find indexes of min_duration events and bin the next year deciles."""
        inds = event_inds(m0, min_duration)
        inds_alt = np.flatnonzero(m1)
        # Number of years between dry/wet event and the next wet/dry year
        if inds_alt.size == 0:
            k = np.zeros(len(bins) - 1, dtype=int)
            return k
        max_alt_ind = inds_alt.max()
        n_years = np.array(
            [
                inds_alt[inds_alt > i][0] - (i + min_duration - 1)
                for i in inds
                if i < max_alt_ind
            ]
        )
        k, _ = np.histogram(n_years, bins=bins)
        return k

    # Create decile threshold mask
    m0 = decile <= 3
    m1 = decile >= 8
    if transition_from == "wet":
        m0, m1 = m1, m0

    # Bins for transition years
    max_years = 20  # Maximum duration (ensures consistent output size)
    # Note that years[0] is the first year after the dry/wet event
    bins = np.arange(max_years + 1, dtype=int)

    k = xr.apply_ufunc(
        transition_years,
        m0,
        m1,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[["years"]],
        vectorize=True,
        # dask="parallelized",
        kwargs=dict(min_duration=min_duration, bins=bins),
        output_dtypes=["int"],
    )

    return k, bins


def binom_ci(n, p=0.3):
    """Apply binomial test to calculate p-values."""
    ci0, ci1 = xr.apply_ufunc(
        scipy.stats.binom.interval,
        0.95,
        n,
        # p,
        input_core_dims=[[], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(p=p),
        output_dtypes=["float64", "float64"],
    )
    return ci0, ci1


def apply_binomtest(k, n, p=0.3, alternative="two-sided"):
    """Apply binomial test to calculate p-values (unused - delete?)."""

    def bionomtest_pvalue(k, n, **kwargs):
        if n > 0:
            pvalue = scipy.stats.binomtest(k, n, **kwargs).pvalue
        else:
            pvalue = np.nan
        return pvalue

    p = xr.apply_ufunc(
        bionomtest_pvalue,
        k,
        n,
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(p=p, alternative=alternative),
        output_dtypes=["float64"],
    )
    return p


def plot_binom(n=35):
    """Simple plot of confidence intervals from the binomial test (delete?)."""
    x = np.arange(n)
    tail = ["greater", "less", "two-sided"]
    pvalues = []
    for alternative in tail:
        pvalues.append(
            [scipy.stats.binomtest(i, n, 0.3, alternative).pvalue for i in x]
        )
    interval = list(scipy.stats.binom.interval(0.95, n, p=0.3))

    plt.figure(figsize=(8, 5))
    plt.title("Bionomtest for n={} (95% CI: {:.0f}-{:.0f})".format(n, *interval))
    for p, c, label in zip(pvalues, ["b", "r", "g"], tail):
        plt.plot(x, p, c=c, label=label)

    for i in range(2):
        plt.axhline([0.05, 0.95][i], c="k")
        plt.axvline(interval[i], c="k")

    plt.margins(x=0, y=1e-2)
    plt.ylabel("p-value")
    plt.xlabel("Number of successes (k)")
    plt.legend(title="Alternative")
