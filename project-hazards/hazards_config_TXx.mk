# Configuration file for UNSEEN hazards

PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/hazards

## Metric calculation

VAR=tasmax
UNITS=degC

METRIC_OPTIONS_FCST=--lat_bnds -46 -6 --lon_bnds 110 157 --output_chunks lead_time=50 --reset_times
METRIC_OPTIONS=--variables ${VAR} --time_freq YE-JUN --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units tasmax=${UNITS}

## Labels
METRIC=txx
REGION=aus
TIMESCALE=annual-jul-to-jun
METRIC_PLOT_LABEL='Temperature [°C]'
METRIC_PLOT_UPPER_LIMIT=60

## Function options
INDEPENDENCE_OPTIONS=--confidence_interval 0.99 --n_resamples 1000
MIN_LEAD_OPTIONS=--min_lead_kwargs variables=min_lead shapefile=/g/data/xv83/unseen-projects/outputs/hazards/shapefiles/australia.shp shape_overlap=0.1 spatial_agg=median