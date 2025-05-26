# Configuration file for UNSEEN project on bias (rx1day)

PROJECT_NAME=bias
ENV_DIR=/g/data/xv83/dbi599/miniconda3/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/bias

## Labels
METRIC=rx1day
REGION=AUS300i
TIMESCALE=annual-jul-to-jun

## Metric calculation
VAR=pr
UNITS=mm day-1
TIME_FREQ=YE-JUN
METRIC_OPTIONS=--variables ${VAR} --time_freq ${TIME_FREQ} --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units ${VAR}='${UNITS}' --regrid_name ${REGION} --regrid_lat_offset 1
METRIC_OPTIONS_FCST= --output_chunks lead_time=50 --reset_times



