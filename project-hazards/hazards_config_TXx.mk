# Configuration file for UNSEEN hazards

ENV_DIR=/g/data/xv83/as3189/conda/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/hazards
FIG_DIR=${PROJECT_DIR}/figures/txx
## Notebook options
METRIC_PLOT_LABEL='Temperature [°C]'
METRIC_PLOT_UPPER_LIMIT=60
BIAS_CORRECTION=additive
SIMILARITY_TEST=ks
TIME_AGG=maximum
covariate_period=1960-2020 # todo: define period for the covariate here?
SHAPE_OVERLAP=0.1
SHAPEFILE=/g/data/xv83/unseen-projects/outputs/hazards/shapefiles/australia.shp
MIN_LEAD_SHAPE_SPATIAL_AGG=median
TIME_FREQ=YE-JUN

## Metric calculation
VAR=tasmax
UNITS=degC
METRIC_OPTIONS_MODEL=--lat_bnds -46 -6 --lon_bnds 110 157 --output_chunks lead_time=50 --reset_times
METRIC_OPTIONS=--variables ${VAR} --time_freq ${TIME_FREQ} --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units tasmax=${UNITS}

## Labels
METRIC=txx
REGION=aus
TIMESCALE=annual-jul-to-jun



## Independence test options
INDEPENDENCE_OPTIONS=--confidence_interval 0.99 --n_resamples 1000
## Minimum lead time file options (independence file kwargs e.g., median min_lead over shapefile)
MIN_IND_LEAD_OPTIONS=--min_lead_kwargs variables=min_lead shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP} spatial_agg=${MIN_LEAD_SHAPE_SPATIAL_AGG}

## GEV distribution options
FITSTART=LMM
GEV_BEST_MODEL_TEST=bic
GEV_STATIONARY_OPTIONS=--fitstart ${FITSTART} --retry_fit --assert_good_fit --file_kwargs variables=tasmax shapefile=${SHAPEFILE} shape_overlap=0.1
GEV_NONSTATIONARY_OPTIONS=--covariate "time.year" --fitstart ${FITSTART} --retry_fit --pick_best_model ${GEV_BEST_MODEL_TEST} --file_kwargs variables=tasmax shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP}

