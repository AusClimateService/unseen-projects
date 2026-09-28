# Configuration file for wcrp-txx analysis (climatology files)

PROJECT_NAME=wcrp-txx
ENV_DIR=/g/data/xv83/dbi599/miniconda3/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/wcrp-txx

## Labels
METRIC=psl-june-july
REGION=north-america
TIMESCALE=annual

## Metric calculation
VAR=psl
UNITS=hPa
TIME_FREQ=YE-DEC
METRIC_OPTIONS=--variables ${VAR} --lat_bnds 25 75 --lon_bnds 175 305 --months 6 7 --time_freq ${TIME_FREQ} --time_agg mean --input_freq D --time_agg_min_tsteps 59 --units ${VAR}='${UNITS}' 
METRIC_OPTIONS_FCST= --output_chunks lead_time=50

