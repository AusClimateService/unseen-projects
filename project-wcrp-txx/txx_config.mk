# Configuration file for wcrp-txx analysis

PROJECT_NAME=wcrp-txx
ENV_DIR=/g/data/xv83/dbi599/miniconda3/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/wcrp-txx

## Labels
METRIC=txx
REGION=PNW
TIMESCALE=annual

## Metric calculation
VAR=tasmax
UNITS=degC
TIME_FREQ=YE-DEC
METRIC_OPTIONS=--variables ${VAR} --lat_bnds 45 52 --lon_bnds 119 123 --spatial_agg weighted_mean --time_freq ${TIME_FREQ} --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units ${VAR}='${UNITS}' 
METRIC_OPTIONS_FCST= --output_chunks lead_time=50 --reset_times
