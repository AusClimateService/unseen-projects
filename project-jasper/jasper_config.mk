# Configuration file for TC Jasper analysis

PROJECT_DIR=/g/data/xv83/dbi599/unseen-projects/tc-jasper

VAR=pr
METRIC=rx5day
REGION=daintree-river
TIMESCALE=annual-aug-to-sep
SHAPEFILE=/g/data/xv83/dbi599/unseen-projects/tc-jasper/shapefiles/daintree_river.shp
OBS_DATASET=AGCD-CSIRO

METRIC_OPTIONS_FCST=--lat_bnds -20 -10 --lon_bnds 140 150 --shp_overlap 0.1 --output_chunks lead_time=50
METRIC_OPTIONS=--variables ${VAR} --spatial_agg weighted_mean --rolling_sum_window 5 --shapefile ${SHAPEFILE} --time_freq A-AUG --time_agg max --input_freq D --units_timing middle --reset_times --complete_time_agg_periods --verbose --time_agg_dates --units pr='mm day-1'







