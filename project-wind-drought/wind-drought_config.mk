# Configuration file for wheatbelt analysis

PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/wind-drought

## Metric calculation

VAR=sfcWind
REGION=swis
# nem-2030 se-2030 swis nwis
SHAPEFILE=${PROJECT_DIR}/shapefiles/${REGION}.shp
SHAPE_OVERLAP=0.1

METRIC_OPTIONS_FCST=--output_chunks lead_time=50 
METRIC_OPTIONS=--variables ${VAR} --months 5 6 7 --shapefile ${SHAPEFILE} --shp_overlap ${SHAPE_OVERLAP} --spatial_agg weighted_mean --verbose --lat_bnds -45 -10 --lon_bnds 110 160

## Labels

METRIC=sfcWind
TIMESCALE=MJJ







