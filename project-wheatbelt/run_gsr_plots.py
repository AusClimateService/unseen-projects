"""Run low/high growing season (Apr-Oct) rainfall event plots for AGCD and DCPP models."""

import numpy as np
import xarray as xr

from process_gsr_data import (
    home,
    names,
    models,
    gsr_data_aus_AGCD,
    gsr_data_aus_DCPP,
    gsr_data_regions,
)
from gsr_events import (
    Events,
    get_events_au,
)
import plot_gsr_regions
import plot_gsr_maps
from utils import combine_all_figures

n_samples = 10000

for model in names:
    for operator in ["less", "greater"]:
        time = "time" if model == "AGCD" else "lead_time"
        event = Events(n=3, operator=operator, minimize=True)
        # Properties of events with no upper limit (for maximum duration plot)
        event_max = Events(2, operator, minimize=False, fixed_duration=False)
        events = [Events(i, operator, minimize=j) for i in [1, 2, 3] for j in [1, 0]]

        # Region specific plots
        dv = gsr_data_regions(model, plot_gsr_regions.regions)
        if model != "AGCD":
            plot_gsr_regions.plot_timeseries_DCPP(dv, event, model)
            plot_gsr_regions.plot_transition_histogram_downsampled(
                dv.tercile, model, event, n_samples
            )
            plot_gsr_regions.plot_transition_matrix_downsampled(
                dv.tercile, model, event
            )
            plot_gsr_regions.plot_transition_sample_size(
                dv.tercile, model, event, n_samples
            )
            plot_gsr_regions.plot_duration_histogram_downsampled(
                model, models, event, time, n_samples
            )
        else:
            for ev in [event_max, *events]:
                plot_gsr_regions.plot_timeseries_AGCD(dv, ev)

        plot_gsr_regions.plot_duration_histogram(dv, event_max, model, time)
        plot_gsr_regions.plot_transition_histogram(
            dv.decile, model, event, time, var="decile"
        )
        plot_gsr_regions.plot_transition_histogram(
            dv.decile, model, event, time, var="binned_decile"
        )
        plot_gsr_regions.plot_transition_histogram(
            dv.tercile, model, event, time, var="tercile"
        )
        plot_gsr_regions.plot_transition_pie_chart(dv.tercile, model, event, time)
        plot_gsr_regions.plot_transition_probability_matrix(
            dv.tercile, model, event, time
        )
        plot_gsr_regions.plot_transition_duration_histogram(
            dv.decile, model, event, time
        )

        # Map plots
        if model == "AGCD":
            dv = gsr_data_aus_AGCD()
            n_times = dv.pr.time.count("time").load().item()
        else:
            dv = gsr_data_aus_DCPP(model)
            n_times = (~np.isnan(dv.decile)).sum(["init_date", "ensemble", "lead_time"])
            n_times = n_times.max().load().item()

        ds_max = get_events_au(dv.pr, dv.decile, event_max, model, time)

        # Stack n=2,3 year event property datasets along dimension "n"
        evs = [Events(i, operator=operator, minimize=True) for i in [2, 3]]
        ds = [get_events_au(dv.pr, dv.decile, evs[i], model, time) for i in [0, 1]]
        ds = xr.concat([ds[i].assign_coords({"n": i + 2}) for i in [0, 1]], dim="n")

        if model != "AGCD":  # Stack DCPP model event properties
            ds = ds.rename({"event": "tmp"})
            ds = ds.stack({"event": ["init_date", "ensemble", "tmp"]})

        plot_gsr_maps.plot_deciles(dv, model)

        # Plot spatial maps for each n-year event duration
        for i, event in enumerate(evs):
            plot_gsr_maps.plot_event_count(
                ds.isel(n=i, drop=True), dv.decile, model, event, n_times
            )
            # Plot frequency n-year events (# of events per 100 years)
            plot_gsr_maps.plot_frequency(ds.isel(n=i, drop=True), model, event, n_times)
            # Maps of minimum, median, and maximum GSR during events
            plot_gsr_maps.plot_event_stats(ds.isel(n=i, drop=True), model, event)

        # Median and maximum duration of consecutive years of high/low GSR
        plot_gsr_maps.plot_duration(ds_max, model, event_max)

        # Plot transition probability map for 1-year decile events
        plot_gsr_maps.plot_persistance_probability(
            dv.tercile, Events(1, operator), model, time
        )

        # Transition probability maps for n-year events (dry, medium & wet transition maps)
        for event in [Events(i + 1, operator) for i in range(3)]:
            plot_gsr_maps.plot_transition_probability(dv.tercile, event, model, time)

for event in [Events(operator="less"), Events(operator="greater")]:
    plot_gsr_regions.plot_transition_histogram_downsampled_all_models(event, n_samples)
    plot_gsr_regions.plot_downsampled_duration_histogram(
        "all", models, event, n_samples
    )

combine_all_figures(home)
