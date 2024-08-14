# Configuration file for UNSEEN hazards

PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/hazards

## Metric calculation

VAR=tasmax

METRIC_OPTIONS_FCST=--lat_bnds -50 -5 --lon_bnds 110 157 --output_chunks lead_time=50
METRIC_OPTIONS=--variables ${VAR} --time_freq YE-JUN --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units tasmax='degC' --verbose 

## Labels

METRIC=txx
REGION=aus
TIMESCALE=annual-jul-to-jun
METRIC_PLOT_LABEL='Temperature [°C]'
METRIC_PLOT_UPPER_LIMIT=55
