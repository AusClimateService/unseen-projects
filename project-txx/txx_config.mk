# Configuration file for UNSEEN project on TXx

PROJECT_NAME=txx
ENV_DIR=/g/data/xv83/as3189/conda/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/${PROJECT_NAME}
NOTEBOOK_IN_DIR=/g/data/xv83/unseen-projects/code
NOTEBOOK_OUT_DIR=/g/data/xv83/unseen-projects/code/project-${PROJECT_NAME}

## Metric calculation
VAR=tasmax
UNITS=degC
TIME_FREQ=YE-JUN
METRIC_OPTIONS=--variables ${VAR} --time_freq ${TIME_FREQ} --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --units ${VAR}='${UNITS}'
# Limit the number of lead times to 9 (fix for DCPP EC-Earth3 members with 11 years of data)
METRIC_OPTIONS_FCST=--lat_bnds -46 -6 --lon_bnds 110 157 --output_chunks lead_time=50 --reset_times --lead_dim_max_size 9 --verbose
REFERENCE_TIME_PERIOD=1961-06-30 2023-06-30

## Labels
METRIC=txx
REGION=aus
TIMESCALE=annual-jul-to-jun
SIMILARITY_TEST=ks

# Shapefile for GEV parameters (mask invalid ocean points to avoid errors)
SHAPE_OVERLAP=0.1
SHAPEFILE=${PROJECT_DIR}/shapefiles/australia.shp

## Independence test options
INDEPENDENCE_OPTIONS=--confidence_interval 0.99 --n_resamples 1000
## Minimum lead time file options (independence file kwargs e.g., median min_lead over shapefile)
MIN_IND_LEAD_OPTIONS=--min_lead_kwargs variables=min_lead shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP} spatial_agg=${MIN_LEAD_SHAPE_SPATIAL_AGG}

## GEV distribution options
FITSTART=LMM
GEV_TEST=bic
GEV_STATIONARY_OPTIONS=--fitstart ${FITSTART} --use_basinhopping --assert_good_fit --file_kwargs variables=${VAR} shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP}
GEV_NONSTATIONARY_OPTIONS=--covariate "time.year" --fitstart ${FITSTART} --use_basinhopping --file_kwargs variables=${VAR} shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP}
GEV_OBS_OPTIONS=--reference_time_period ${REFERENCE_TIME_PERIOD}

## Notebook options
MIN_LEAD_SHAPE_SPATIAL_AGG=median
TIME_AGG=maximum
# Non-stationary GEV covariate used for return levels
COVARIATE_BASE=2025
# Period for trend calculation (string converted to python)
GEV_TREND_PERIOD='[1961, 2025]'
# Dictionary of plot options for spatial analysis notebook (string converted to python)
PLOT_DICT='dict(metric="TXx", var="${VAR}", var_name="Temperature", units="°C", units_label="Temperature [°C]", freq="${TIME_FREQ}", cmap=cmap_dict["ipcc_temp_seq"], cmap_anom=plt.cm.RdBu_r, ticks=np.arange(24, 58 + 2, 2), ticks_anom=np.arange(-8.5, 9.5, 1), ticks_anom_std=np.arange(-8.5, 9.5, 1), ticks_anom_pct=np.arange(-29, 31, 2), ticks_anom_ratio=np.arange(0.7, 1.35, 0.05), ticks_trend=np.arange(-4.75, 5, 0.5), ticks_param_trend={"location": np.arange(-0.6, 0.61, 0.1), "scale": np.arange(-0.12, 0.122, 0.02)}, cbar_extend="both", agcd_mask=False)'
#  Plot additive/multiplicative bias corrected metric in spatial analysis notebook (True/False)#
PLOT_ADDITIVE_BC=1
PLOT_MULTIPLICATIVE_BC=0
