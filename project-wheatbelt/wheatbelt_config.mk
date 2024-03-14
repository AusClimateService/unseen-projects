# Configuration file for wheatbelt analysis

PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/wheatbelt

## Metric calculation

VAR=pr
SHAPEFILE=${PROJECT_DIR}/shapefiles/crops_WA.shp
SHAPE_OVERLAP=0.01

METRIC_OPTIONS_FCST=--lat_bnds -40 -10 --lon_bnds 110 150 --shp_overlap ${SHAPE_OVERLAP} --output_chunks lead_time=50
METRIC_OPTIONS=--variables ${VAR} --spatial_agg weighted_mean --shapefile ${SHAPEFILE} --months 4 5 6 7 8 9 10 --time_freq A-DEC --time_agg sum --time_agg_min_tsteps 213 --input_freq D --verbose --units pr='mm day-1' 

## Labels

METRIC=growing-season-pr
REGION=crops-wa
TIMESCALE=AMJJASO
METRIC_PLOT_LABEL='total AMJJASO rainfall (mm)'
METRIC_PLOT_UPPER_LIMIT=1000







