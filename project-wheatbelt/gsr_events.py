"""Low/high growing season (Apr-Oct) rainfall event analysis.

- Define events with consecutive years that meet a threshold (no overlapping years)
- Calculate transition probabilities (overlapping years)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy
import xarray as xr

from process_gsr_data import gsr_data_regions


@dataclass
class Events:
    """Stores event metadata used in figure titles and labels."""

    def __init__(
        self,
        n: int = 3,
        operator: str = "less",
        fixed_duration: bool = True,
    ):
        """Initialise event metadata."""
        # Number of event years / event minimum duration
        self.n = n
        self.min_duration = n
        # Operator to apply a threshold (less/greater than)
        self.operator = operator
        # Define events that are always min_duration long
        self.fixed_duration = fixed_duration

        # Figure labels
        if self.operator == "less":
            self.type = "LGSR"
            self.threshold = 1  # 3
            self.sym = "≤"
            self.alt_name = "dry"
            self.name = "low"
        else:
            self.type = "HGSR"
            self.threshold = 3  # 8
            self.sym = "≥"
            self.alt_name = "wet"
            self.name = "high"

        self.decile = f"{self.sym}{self.threshold} decile"
        self.tercile = f"{self.sym}{self.threshold} tercile"


def get_events_1d(
    da,
    threshold,
    min_duration,
    operator="less",
    fixed_duration=True,
):
    """Label contiguous regions of da <=/=> threshold with duration >= min_duration.

    Parameters
    ----------
    da : xr.DataArray
        1D array of data values
    threshold : int
        Threshold for an event
    min_duration : int
        Minimum duration of event
    operator : {"less", "greater"}, optional
        Operator to apply a threshold
    fixed_duration : bool, optional
        Define events that are always min_duration long

    Notes
    -----
    - Calculates events in which there are a minimum number of values in
    a row at or below/above a threshold (no overlapping years)
    - Used for plots of frequency, avg rainfall & max duration of events

    Example
    -------
    - Find events of duration >= min_duration:
        get_events_1d(d, t, min_duration, fixed_duration=False)
    - Find events of duration == min_duration:
        get_events_1d(d, t, min_duration, fixed_duration=True)
    """

    assert operator in ["less", "greater"], "Operator must be 'less' or 'greater'"
    assert min_duration >= 1, "min_duration must be >= 1"
    assert isinstance(min_duration, int)
    assert da.ndim == 1, "Input data must be 1D"
    if np.isnan(da).all():
        # Return NaNs if all input data is NaN
        return da * np.nan

    # Threshold mask
    if operator == "less":
        da_mask = da <= threshold
    else:
        da_mask = da >= threshold

    # Find contiguous regions of da_mask == True
    # da => events: assigns ID for each contiguous region
    events_orig, n_events = scipy.ndimage.label(da_mask)

    # This will be updated to remove event IDs that do not meet criteria
    events = events_orig.copy()

    # Iterate through each event ID to check event duration criteria
    # (NB `ev` only corresponds to `events_orig`)
    for ev in range(1, n_events + 1):
        # Indices of values within the event
        inds = np.nonzero(events_orig == ev)

        # Count number of unmasked elements
        duration = np.count_nonzero(inds)  # events_orig[inds].count()

        # Enforce event lower limit (and upper limit if requested)
        if duration < min_duration or (fixed_duration and duration != min_duration):
            # Delete event ID
            events[inds] = 0
            # Roll back following event IDs
            events[np.nonzero(events_orig > ev)] -= 1
            continue

        if fixed_duration:
            assert len(events_orig[inds]) == min_duration

    return events


def get_event_properties_1d(
    data,
    quantile,
    events,
    times,
    n_events,
    variables,
):
    """Get consecutive year event properties .

    Parameters
    ----------
    data : array-like
        Rainfall timeseries
    quantile : array-like
        Quantile timeseries
    events : xr.DataArray
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

    Notes
    -----
    - This function is overly complicated to work with xarray.apply_ufunc
    (doesn't like mixed output sizes / datasets).
    See `get_gsr_events` for vectorized version
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
            ds["quantile_next"][loc] = quantile[inds[-1] + 1]  # Next year quantile
        ds["total_samples"][loc] = len(data[~np.isnan(data)])

    return tuple([ds[v] for v in variables])


def get_gsr_events(da, da_quantile, time_dim="time", **kwargs):
    """Get events & event properties

    This function vectorizes `get_events_1d` & `get_event_properties_1d`.

    Parameters
    ----------
    da : xr.DataArray
        Data values (e.g., rainfall)
    da_quantile : xr.DataArray
        Data converted to quantiles
    time_dim : str, default "time"
        Name of the core time dimension in which to search for events
    **kwargs : dict
        Additional arguments to pass to `get_events_1d`

    Returns
    -------
    events : xr.DataArray
        Time series of labeled events
    ds : xr.Dataset
        Dataset of event properties

    Example
    -------
    - Find 3-year low tercile events in DCPP model data:
    events, ds = get_gsr_events(
        da,
        da_quantile,
        time_dim="lead_time",
        threshold=1,
        min_duration=3,
        operator="less",
        fixed_duration=True,
    )

    Notes
    -----
    - da and da_quantile must have dim 'time' (but searches events along time_dim)
    - Super tedious workaround to get around xarray.apply_ufunc limitations
    with mixed output sizes / datasets.
    """

    # Get time series of labeled events
    events = xr.apply_ufunc(
        get_events_1d,
        da_quantile,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        kwargs=kwargs,
        output_dtypes=["float64"],
    )

    # List of event properties to calculate (expected output variables)
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
        "quantile_next",
        "total_samples",
    ]

    # Max number of events (for length of dimension)
    n_events = events.max().load().item()  # Max event ID

    # Convert time to numeric if datetime64[ns] (avoids issues with dask)
    if da["time"].dtype in ["datetime64[ns]"]:
        epoch = np.datetime64("1900-01-01T00:00:00")
        times = (da["time"] - epoch) / np.timedelta64(1, "D")
    else:
        # Assumes "time" is the actual time variable
        times = da["time"]

    # Define output dtypes for event properties
    dtypes = []
    for v in variables:
        dtype = "float64" if "time" not in v else str(times.dtype)
        dtypes.append(dtype)

    da_list = xr.apply_ufunc(
        get_event_properties_1d,
        da.chunk({time_dim: -1}),
        da_quantile.chunk({time_dim: -1}),
        events,
        times.chunk({time_dim: -1}),
        input_core_dims=[[time_dim], [time_dim], [time_dim], [time_dim]],
        output_core_dims=[["event"] for _ in range(len(variables))],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(n_events=n_events, variables=variables),
        output_dtypes=dtypes,
        dask_gufunc_kwargs={"output_sizes": {"event": n_events}},
    )

    # Create dataset from output DataArrays
    ds = xr.Dataset(coords={"event": np.arange(n_events)})
    for v, dv in zip(variables, da_list):
        if "time" in v and da["time"].dtype == "datetime64[ns]":
            # Convert times back to datetime64[ns]
            ds[v] = dv * np.timedelta64(1, "D") + epoch
        else:
            ds[v] = dv

    ds["total_samples"] = ds["total_samples"].isel(
        event=0, drop=True
    )  # Duplicated for each event for consistent output

    return events, ds


def get_gsr_events_gridded(da, da_quantile, time_dim="time", **kwargs):
    """Get GSR event property dataset for all grid points.

    This function fixes some issues in `get_gsr_events` when using gridded data.

    Parameters
    ----------
    da : xr.DataArray
        Data values (e.g., rainfall)
    da_quantile : xr.DataArray
        Data converted to quantiles
    time_dim : str, default "time"
        Name of the core time dimension in which to search for events
    **kwargs : dict
        Additional arguments to pass to `get_gsr_events`

    Returns
    -------
    ds : xr.Dataset
        Dataset of event properties
    """

    def convert_time(time_start, time_end):
        """Convert time_start and time_end to datetime64[ns]."""

        if not isinstance(time_start, (float, int)) and pd.notnull(time_start):
            time_start = np.datetime64(time_start.isoformat())
            time_end = np.datetime64(time_end.isoformat())
        else:
            time_end = pd.NaT
            time_start = pd.NaT
        return time_start, time_end

    _, ds = get_gsr_events(da, da_quantile, time_dim=time_dim, **kwargs)

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

    # Check event durations meets criteria
    duration = ds.duration.values
    if kwargs.get("fixed_duration", False):
        assert np.all(duration == kwargs["min_duration"], where=~np.isnan(duration))
    else:
        assert np.all(duration >= kwargs["min_duration"], where=~np.isnan(duration))

    return ds


def run_start_inds_1d(da_mask, run_duration):
    """Get indexes where a run of consecutive begins.

    Parameters
    ----------
    da_mask : array-like
        Boolean mask of sample above/ below threshold (1D timeseries)
    run_duration : int
        Number of consecutive elements in run

    Returns
    -------
    inds : array-like
        Indexes where a run of consecutive elements starts

    Notes
    -----
    - Overlapping runs are allowed (unlike get_gsr_events)
    - Runs has length of exactly `run_duration`
    - Only implemented for min_duration > 3
    - Forces loading into memory
    """

    assert da_mask.ndim == 1, "Input data must be 1D"
    assert run_duration in [1, 2, 3], "run_duration > 3 not implemented"

    if isinstance(da_mask, xr.DataArray):
        da_mask = da_mask.values  # Only works for numpy arrays

    if run_duration == 1:
        # Return all indexes where True
        inds = np.flatnonzero(da_mask)

    elif run_duration == 2:
        # Find inds where the i and i+1 elements match
        inds = np.flatnonzero(da_mask[:-1] & da_mask[1:])
        # Drop consecutive "False" events (not needed here?)
        inds = inds[da_mask[inds]]

    elif run_duration == 3:
        # Find inds where the i,i+1,i+2 elements match
        inds = np.flatnonzero(da_mask[:-2] & da_mask[1:-1] & da_mask[2:])
        # Drop consecutive "False" events
        inds = inds[da_mask[inds]]

    return inds


def get_run_next_values_1d(m, da, min_duration):
    """Find indexes of min_duration events and bin the next year values.

    Parameters
    ----------
    m : array-like
        Boolean mask of sample above/ below threshold
    da : array-like
        Data values
    min_duration : int
        Minimum duration of event

    Returns
    -------
    k : array-like
        Next year data values after events (NaN if no event)
    n : int
        Total number of next year values
    """

    inds = run_start_inds_1d(m, min_duration)

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

    # Drop any transitions to NaN
    da_next = da_next[~np.isnan(da_next)]

    # Total number of next year values
    total = da_next.size

    # Assign the next year values to the correct indexes
    k[: len(da_next)] = da_next

    return k, total


def transition_probability(
    da,
    threshold,
    operator,
    min_duration,
    var="tercile",
    time_dim="time",
    binned=True,
):
    """Calculate the probability of transitioning to another year of a low/high quantile.

    Parameters
    ----------
    da : xr.DataArray
        Data converted to quantiles
    threshold : float
        Quantile threshold
    operator : {"less", "greater"}
        Operator to apply a threshold
    min_duration : int or array-like
        Minimum duration of event
    var : {"decile", "binned_decile", "tercile"}, optional
        Variable to bin, by default "tercile" ()
    time_dim : str, optional
        Name of the time dimension, by default "time"
    binned : bool, optional
        Bin the quantiles into dry/medium/wet or 1-10, by default True

    Returns
    -------
    k : float or xr.DataArray
        Number of next year quantiles (q=dry, medium, wet)
    n : float or xr.DataArray
        Total number of next year quantiles
    bins : array-like
        Bin edges of output

    Notes
    -----
    - Calculates the probability of transitioning from n years in a row
    above/below a quantile threshold to other quantiles (i.e., dry, medium, wet).
    - Note that this includes overlapping years (unlike get_gsr_events).
    - If the last year in the series meets the criteria, it is dropped from
    the total event count next year because the 'next year' is not known.
    - Used for persistance_probability, transitions_probability and
    transition_matrix plots.
    """

    assert np.all(min_duration <= 3)  # requires at least 2 years of data after event?
    assert time_dim in da.dims, f"Time {time_dim} dimension not found in data array"

    # Create quantile threshold mask
    if operator == "less":
        m = da <= threshold
    else:
        m = da >= threshold

    if isinstance(min_duration, int):
        min_duration = np.array([min_duration])
    if isinstance(min_duration, np.ndarray):
        min_duration = xr.DataArray(min_duration, dims="n", attrs={"n": "n"})

    da_next, n = xr.apply_ufunc(
        get_run_next_values_1d,
        m,
        da,
        min_duration,
        input_core_dims=[[time_dim], [time_dim], []],
        output_core_dims=[[time_dim], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"] * 2,
    )

    if not binned:
        return da_next, n

    elif binned:
        # Create bins for quantiles (dry/medium/wet) or bin each quantile
        if var == "decile":
            bins = np.arange(1, 12)
        elif var == "binned_decile":
            bins = np.array([1, 4, 8, 11])
        elif var == "tercile":
            bins = np.arange(1, 5)

        k, _ = xr.apply_ufunc(
            np.histogram,
            da_next,
            input_core_dims=[[time_dim]],
            output_core_dims=[["q"], ["b"]],
            vectorize=True,
            dask="parallelized",
            kwargs=dict(bins=bins),
            output_dtypes=["int"] * 2,  # ((len(bins) * 2) - 1),
            # output_sizes={"q": len(bins) - 1, "b": len(bins) - 1},
            dask_gufunc_kwargs={
                "output_sizes": {"q": len(bins) - 1, "b": len(bins) - 1}
            },
        )
        return k, n, bins


def downsampled_transition_probability(
    da, event, regions, target="AGCD", n_resamples=10000, var="tercile"
):
    """Resample model transition probabilities to a target sample sizes."""

    assert var == "tercile", "Only tercile variable currently supported"
    rng = np.random.default_rng(seed=0)
    bins = np.arange(1, 5)
    ds = xr.Dataset(coords={"x": regions, "n": np.arange(1, 4), "q": bins[:-1]})

    if target == "AGCD":
        # AGCD sample sizes for each region
        dv_agcd = gsr_data_regions("AGCD")
        k_agcd, n_agcd, _ = transition_probability(
            dv_agcd.tercile,
            event.threshold,
            event.operator,
            np.arange(1, 4),
            var=var,
            time_dim="time",
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
        np.zeros((len(regions), 3, 3, n_resamples)), dims=("x", "n", "q", "sample")
    )

    # Get all of the next year terciles
    dx_next, _ = transition_probability(
        da,
        event.threshold,
        event.operator,
        np.arange(1, 4),  # event durations (>3 not implemented)
        var=var,
        time_dim="lead_time",
        binned=False,
    )
    dx_next_stacked = dx_next.stack(dict(sample=["ensemble", "init_date", "lead_time"]))

    for i in range(2):
        for j in range(3):
            # Drop NaNs
            dx = dx_next_stacked.isel(n=j, x=i).dropna("sample", how="all")
            # Draw random samples (n_resamples of size agcd_sample_sizes)
            dx_sampled = rng.choice(
                dx, (ds.total.isel(n=j, x=i).load().item(), n_resamples), replace=True
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


def get_event_duration_counts(
    event, ds, m, density, downsample=True, n_resamples=None, N_obs=None
):
    """Get counts of event durations for a specific dataset and region.

    For models other than AGCD, downsample to match number of samples in AGCD data.

    Parameters
    ----------
    event : Events
        Event metadata.
    ds : xarray.Dataset
        Dataset containing data var and quantile data arrays at a single region.
    m : str
        Model name.
    density : bool
        If True, convert counts to probability (%).
    downsample : bool
        If True, downsample model data to match N_obs sample size.
    n_resamples : int, optional
        Number of resamples for downsampling (required for models other than AGCD).
    N_obs : int, optional
        Number of samples in AGCD data (required for models other than AGCD).

    Returns
    -------
    counts : array-like
        Counts of event durations in each bin.
    bins : array-like
        Histogram bin edges.
    N : int
        Total number of samples used to calculate counts.
    """

    assert (
        "x" not in ds.pr.dims
    ), "Dataset must be for a single region (no 'x' dimension)."

    min_duration = 1
    # Plot durations up to 9 years
    bins = np.arange(min_duration, 11)
    time_dim = "lead_time" if m != "AGCD" else "time"

    _, ds_events = get_gsr_events(
        ds.pr,
        ds.tercile,
        threshold=event.threshold,
        min_duration=min_duration,
        operator=event.operator,
        fixed_duration=False,
        time_dim=time_dim,
    )

    if m == "AGCD":
        N = ds[time_dim].size
        counts, _ = np.histogram(ds_events.duration, bins=bins)

    elif not downsample:
        # No downsampling, use all model data
        ds_events = ds_events.stack(dict(sample=["ensemble", "init_date"]))
        counts, _ = np.histogram(ds_events.duration, bins=bins)

        # Get number of samples
        ds_stacked = ds.stack(dict(sample=["ensemble", "init_date", "lead_time"]))
        N = ds_stacked.dropna("sample", how="all").sample.size

    else:
        # Get n_resamples of model subsamples
        # Approximate number of obs samples (nb "events" must calculated over lead_time dim)
        ds_events = ds_events.stack(dict(sample=["ensemble", "init_date"]))
        # Get N samples of subsampled data
        # len(subsample) = n_init_blocks * len(lead_time) (e.g., 12x10-year runs)
        n_lead = ds[time_dim].size
        n_init_blocks = N_obs // n_lead
        n_init_blocks = int(round(N_obs / n_lead, 0))
        N = n_lead * n_init_blocks

        if m == "CAFE":
            # Half of the CAFE runs have one less lead time
            _n_lead = (n_lead + (n_lead - 1)) / 2
            N = _n_lead * n_init_blocks

        # Select n_resamples of subsamples
        rng = np.random.default_rng(seed=42)
        ds_events_sampled = rng.choice(
            ds_events.duration.T,
            (n_resamples, n_init_blocks),
            replace=True,
            axis=0,
        )
        ds_events_sampled = xr.DataArray(
            ds_events_sampled, dims=("sample", "block", "event")
        )

        # Stack the events in each subsample (sample, block, event) -> (sample, subsample)
        ds_events_sampled = ds_events_sampled.stack(dict(subsample=["block", "event"]))

        # Bin the durations in each subsample
        counts, _ = xr.apply_ufunc(
            np.histogram,
            ds_events_sampled,
            input_core_dims=[["subsample"]],
            output_core_dims=[["bin"], ["bin_edges"]],
            vectorize=True,
            dask="parallelized",
            kwargs=dict(bins=bins),
            output_dtypes=["int"] * ((len(bins) * 2) - 1),
        )

    if density:
        # Convert from counts to probability (%)
        counts = (counts / N) * 100
    return counts, bins, N


def transition_time(
    quantiles, min_duration, time_dim="time", transition_from="dry", var="tercile"
):
    """Count transitions between n-year dry/wet events and next high/low year.

    Parameters
    ----------
    quantiles : xr.DataArray
        Data converted to quantiles (e.g., deciles, terciles)
    min_duration : int
        Minimum duration of event
    time_dim : str, optional
        Name of the time dimension, by default "time"
    transition_from : {"dry", "wet"}, optional
        Transition from dry or wet events, by default "dry"
    var : {"decile", "tercile"}, optional
        Variable to bin, by default "tercile"

    Returns
    -------
    k : float or xr.DataArray
        Count of years between low/high events
    bins : array-like
        Bin edges of output
    """

    assert min_duration <= 3

    def transition_years(m0, m1, min_duration, bins):
        """Find indexes of min_duration events and bin the next year quantile."""

        inds = run_start_inds_1d(m0, min_duration)
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

    # Create quantiles threshold mask
    if var == "decile":
        m0 = quantiles <= 3
        m1 = quantiles >= 8
    elif var == "tercile":
        m0 = quantiles <= 1
        m1 = quantiles >= 3
    if transition_from == "wet":
        m0, m1 = m1, m0

    # Bins for transition years
    max_years = 20  # Maximum duration (ensures consistent output size)

    # Note that years[0] is the first year after the dry/wet event
    bins = np.arange(max_years + 1, dtype=int)

    k = xr.apply_ufunc(
        transition_years,
        m0.compute(),
        m1.compute(),
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[["years"]],
        vectorize=True,
        # dask="parallelized",
        kwargs=dict(min_duration=min_duration, bins=bins),
        output_dtypes=["int"],
    )

    return k, bins


def binom_ci(n, p=1 / 3, confidence_level=0.95):
    """Apply binomial test to determine confidence intervals.

    Parameters
    ----------
    n : int or xr.DataArray
        Total number of samples
    p : float, optional
        Expected probability, by default 1/3
    confidence_level : float, optional
        Confidence level, by default 0.95

    Returns
    -------
    ci0 : float or xr.DataArray
        Lower confidence interval
    ci1 : float or xr.DataArray
        Upper confidence interval
    """

    assert confidence_level < 1, "Confidence level must be between 0 and 1"
    ci0, ci1 = xr.apply_ufunc(
        scipy.stats.binom.interval,
        confidence_level,
        n,
        input_core_dims=[[], []],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(p=p),
        output_dtypes=["float64", "float64"],
    )
    return ci0, ci1
