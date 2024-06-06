"""Low/high growing season (Apr-Oct) rainfall event analysis.

- Define events with consecutive years that meet a threshold (no overlapping years)
- Calculate transition probabilities (overlapping years)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy
import xarray as xr

from process_gsr_data import home, gsr_data_regions


@dataclass
class Events:
    """Stores event metadata used in figure titles and labels."""

    def __init__(
        self,
        n: int = 3,
        operator: str = "less",
        fixed_duration: bool = True,
        minimize: bool = False,
    ):
        """Initialise event metadata."""
        # Number of event years / event minimum duration
        self.n = n
        self.min_duration = n
        # Operator to apply a threshold (less/greater than)
        self.operator = operator
        # Define events that are always min_duration long
        self.fixed_duration = fixed_duration
        # Event contiguous region based on the lowest/highest total
        self.minimize = minimize

        # Figure labels
        if self.operator == "less":
            self.type = "LGSR"
            self.threshold = 3
            self.sym = "≤"
            self.alt_name = "dry"
            self.name = "low"
        else:
            self.type = "HGSR"
            self.threshold = 8
            self.sym = "≥"
            self.alt_name = "wet"
            self.name = "high"

        if not self.fixed_duration:
            self.type += "_max"

        if not self.minimize and self.fixed_duration:
            self.type += "_first"

        self.decile = f"{self.sym}{self.threshold} decile"
        self.tercile = f"{self.sym}{self.threshold} tercile"


def gsr_events(data, decile, time="time", **kwargs):
    """Get events & event properties (vectorize get_events & get_event_properties)."""

    events = xr.apply_ufunc(
        get_events,
        decile,
        input_core_dims=[[time]],
        output_core_dims=[[time]],
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
        times = (data["time"] - epoch) / np.timedelta64(1, "D")
    else:
        # Assumes "time" is the actual time variable
        times = data["time"]

    dtypes = []
    for v in variables:
        dtype = "float64" if "time" not in v else str(times.dtype)
        dtypes.append(dtype)

    da_list = xr.apply_ufunc(
        get_event_properties,
        data.chunk({time: -1}),
        decile.chunk({time: -1}),
        events,
        times.chunk({time: -1}),
        input_core_dims=[[time], [time], [time], [time]],
        output_core_dims=[["event"] for _ in range(len(variables))],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(n_events=n_events, variables=variables),
        output_dtypes=dtypes,
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
    times,
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
    times : array-like
        Time values
    n_events : int
        Number of events
    variables : list of str
        Event properties to calculate

    Returns
    -------
    list of xarray.Dataset
        DataArrays of event properties (ragged arrays)
    """
    # Create dataset to store event properties
    ds = xr.Dataset(coords={"event": np.arange(n_events)})

    for v in variables:
        if "time" in v:
            dtype = str(times.dtype)
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
        ds["time_start"][loc] = times[inds][0]
        ds["time_end"][loc] = times[inds][-1]

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
                    # Set value of remaining time steps to zero
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


def get_events_au(data, decile, event, model, time="time"):
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

    file_events = home / f"data/{event.type}_{event.n}yr_events_aus_{model}.nc"

    if file_events.exists():
        ds = xr.open_dataset(file_events)
        ds = ds.sel(lat=slice(-52, -23))

    else:
        _, ds = gsr_events(
            data,
            decile,
            time=time,
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


def event_next_values(m, da, min_duration):
    """Find indexes of min_duration events and bin the next year values."""
    inds = event_inds(m, min_duration)

    # Indexes of following years that meet criteria
    inds_next = np.array(list(inds)) + min_duration
    # Drop indexes that are out of bounds
    inds_next = inds_next[inds_next < len(m)]
    k = np.zeros(da.size) * np.nan
    # Return zeros if there are no events
    if inds_next.size == 0:
        return k, 0

    # Get the values of the following years
    da_next = da[inds_next]

    # Total number of next year values
    total = da_next.size

    # Assign the next year values to the correct indexes
    k[: len(inds_next)] = da_next

    return k, total


def transition_probability(
    da,
    threshold,
    operator,
    min_duration,
    var="decile",
    time="time",
    binned=True,
):
    """Calculate the probability of transitioning to another year of a low/high decile/tercile.

    Parameters
    ----------
    da : xa.DataArray
        Deciles or terciles
    threshold : float
        Decile threshold
    operator : {"less", "greater"}
        Operator to apply a threshold
    min_duration : int or array-like
        Minimum duration of event
    var : {"decile", "binned_decile", "tercile"}, optional
        Variable to bin, by default "decile"
    time : str, optional
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
    - Used for persistance_probability, transitions_probability and
    transition_matrix plots.
    """
    assert np.all(min_duration <= 3)

    # Create decile threshold mask
    if operator == "less":
        threshold = threshold if var != "tercile" else 1
        m = da <= threshold
    else:
        threshold = threshold if var != "tercile" else 3
        m = da >= threshold

    if isinstance(min_duration, int):
        min_duration = np.array([min_duration])
    if isinstance(min_duration, np.ndarray):
        min_duration = xr.DataArray(min_duration, dims="n")

    da_next, n = xr.apply_ufunc(
        event_next_values,
        m,
        da,
        min_duration,
        input_core_dims=[[time], [time], []],
        output_core_dims=[[time], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"] * (da[time].size + 1),
    )
    if not binned:
        return da_next, n

    elif binned:
        # Create bins for deciles (dry/medium/wet) or bin each decile/tercile
        if var == "decile":
            bins = np.arange(1, 12)
        elif var == "binned_decile":
            bins = np.array([1, 4, 8, 11])
        elif var == "tercile":
            bins = np.arange(1, 5)

        k, _ = xr.apply_ufunc(
            np.histogram,
            da_next,
            input_core_dims=[[time]],
            output_core_dims=[["q"], ["b"]],
            vectorize=True,
            dask="parallelized",
            kwargs=dict(bins=bins),
            output_dtypes=["int"] * ((len(bins) * 2) - 1),
        )
        return k, n, bins


def downsampled_transition_probability(
    da, event, regions, target="AGCD", n_samples=1000, var="tercile"
):
    """Resample model transition probabilities to a target sample sizes."""
    rng = np.random.default_rng(seed=42)
    bins = np.arange(1, 5)
    ds = xr.Dataset(coords={"x": regions, "n": np.arange(1, 4), "q": bins[:-1]})

    if target == "AGCD":
        # AGCD sample sizes for each region
        dv_agcd = gsr_data_regions("AGCD", regions)
        k_agcd, n_agcd, _ = transition_probability(
            dv_agcd.tercile,
            event.threshold,
            event.operator,
            np.arange(1, 4),
            var=var,
            time="time",
            binned=True,
        )
        ds["total"] = n_agcd.T.astype(dtype=int)

    elif isinstance(target, (np.integer, float)):
        ds["total"] = xr.DataArray(
            np.full((len(regions), 3), target, dtype=int), dims=("x", "n")
        )
    else:
        ds["total"] = target

    ds["k"] = xr.DataArray(
        np.zeros((len(regions), 3, 3, n_samples)), dims=("x", "n", "q", "sample")
    )

    # Get all of the next year terciles
    dx_next, _ = transition_probability(
        da,
        event.threshold,
        event.operator,
        np.arange(1, 4),
        var=var,
        time="lead_time",
        binned=False,
    )
    dx_next_stacked = dx_next.stack(dict(sample=["ensemble", "init_date", "lead_time"]))

    for i in range(2):
        for j in range(3):
            # Drop NaNs
            dx = dx_next_stacked.isel(n=j, x=i).dropna("sample", how="all")
            # Draw random samples (n_samples of size agcd_sample_sizes)
            dx_sampled = rng.choice(
                dx, (ds.total.isel(n=j, x=i).item(), n_samples), replace=True
            )
            dx_sampled = xr.DataArray(dx_sampled, dims=("event", "sample"))

            # Bin the terciles
            ds["k"][dict(x=i, n=j)], _ = xr.apply_ufunc(
                np.histogram,
                dx_sampled,
                input_core_dims=[["event"]],
                output_core_dims=[["q"], ["b"]],
                vectorize=True,
                dask="parallelized",
                kwargs=dict(bins=bins),
                output_dtypes=["int"] * ((len(bins) * 2) - 1),
            )

    ds["p"] = (ds.k / ds.total) * 100
    if target == "AGCD":
        ds["k_agcd"] = k_agcd
    return ds


def transition_time(decile, min_duration, time="time", transition_from="dry"):
    """Count the transition times between n-year dry/wet events and next high/low decile year.

    Parameters
    ----------
    decile : xa.DataArray
        Decile values
    min_duration : int
        Minimum duration of event
    time : str, optional
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
        input_core_dims=[[time], [time]],
        output_core_dims=[["years"]],
        vectorize=True,
        # dask="parallelized",
        kwargs=dict(min_duration=min_duration, bins=bins),
        output_dtypes=["int"],
    )

    return k, bins
