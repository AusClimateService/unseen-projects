include ${PROJECT_DETAILS}
include ${MODEL_DETAILS}
include ${OBS_DETAILS}

DASK_CONFIG=/g/data/xv83/unseen-projects/code/dask_local.yml
ENV_DIR?=/g/data/xv83/unseen-projects/unseen_venv
FIG_DIR?=${PROJECT_DIR}/figures
NOTEBOOK_IN_DIR?=/g/data/xv83/unseen-projects/code
NOTEBOOK_OUT_DIR?=/g/data/xv83/unseen-projects/code\
OBS_LABEL?=${OBS_DATASET}

GEV_TEST?=lrt
FITSTART?=LMM

FILEIO=${ENV_DIR}/bin/fileio
PAPERMILL=${ENV_DIR}/bin/papermill
INDEPENDENCE=${ENV_DIR}/bin/independence
STABILITY=${ENV_DIR}/bin/stability
BIAS_CORRECTION=${ENV_DIR}/bin/bias_correction
SIMILARITY=${ENV_DIR}/bin/similarity
MOMENTS=${ENV_DIR}/bin/moments
EVA=${ENV_DIR}/bin/eva

FCST_DATA=/g/data/xv83/unseen-projects/code/file_lists/${MODEL}_${EXPERIMENT}_${VAR}_files.txt
METRIC_FCST=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
METRIC_OBS=${PROJECT_DIR}/data/${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}.nc
METRIC_FCST_ADDITIVE_BC=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
METRIC_FCST_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
# STABILITY_FILE_MEDIAN=${PROJECT_DIR}/data/stability-test-median_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc # todo
STABILITY_PLOT_EMPIRICAL=${FIG_DIR}/stability-test-empirical_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
STABILITY_PLOT_GEV=${FIG_DIR}/stability-test-gev_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
INDEPENDENCE_FILE=${PROJECT_DIR}/data/independence-test_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
INDEPENDENCE_PLOT=${FIG_DIR}/independence-test_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
SIMILARITY_FILE=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.nc
SIMILARITY_ADDITIVE_BC_FILE=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
SIMILARITY_MULTIPLICATIVE_BC_FILE=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
SIMILARITY_PLOT=${FIG_DIR}/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.png
SIMILARITY_ADDITIVE_BC_PLOT=${FIG_DIR}/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.png
SIMILARITY_MULTIPLICATIVE_BC_PLOT=${FIG_DIR}/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.png
MOMENTS_PLOT=${FIG_DIR}/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.png
MOMENTS_ADDITIVE_BC_PLOT=${FIG_DIR}/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.png
MOMENTS_MULTIPLICATIVE_BC_PLOT=${FIG_DIR}/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.png
NINO_FCST=${PROJECT_DIR}/data/nino34-anomaly_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_base-${BASE_PERIOD_TEXT}.nc
NINO_OBS=${PROJECT_DIR}/data/nino34-anomaly_HadISST_1870-2022_base-1981-2010.nc

# Stationary GEV parameters
GEV_STATIONARY_OBS=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}.nc
GEV_STATIONARY_OBS_DROP_MAX=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}_drop_max.nc
GEV_STATIONARY=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
GEV_STATIONARY_ADDITIVE_BC=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
GEV_STATIONARY_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc

# Nonstationary GEV parameters
GEV_NONSTATIONARY_OBS=${PROJECT_DIR}/data/gev_params_nonstationary_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}.nc
GEV_NONSTATIONARY_OBS_DROP_MAX=${PROJECT_DIR}/data/gev_params_nonstationary_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}_drop_max.nc
GEV_NONSTATIONARY=${PROJECT_DIR}/data/gev_params_nonstationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
GEV_NONSTATIONARY_ADDITIVE_BC=${PROJECT_DIR}/data/gev_params_nonstationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
GEV_NONSTATIONARY_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/gev_params_nonstationary_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc

# Mixed (best of stationary/nonstationary) GEV parameters
GEV_BEST_OBS=${PROJECT_DIR}/data/gev_params_nonstationary_${GEV_TEST}_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}.nc
GEV_BEST_OBS_DROP_MAX=${PROJECT_DIR}/data/gev_params_nonstationary_${GEV_TEST}_${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}_drop_max.nc
GEV_BEST=${PROJECT_DIR}/data/gev_params_nonstationary_${GEV_TEST}_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
GEV_BEST_ADDITIVE_BC=${PROJECT_DIR}/data/gev_params_nonstationary_${GEV_TEST}_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
GEV_BEST_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/gev_params_${GEV_TEST}_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc

## print all local variables (for importing variables in py scripts)
print_file_vars :
	$(foreach v, $(.VARIABLES), $(if $(filter file,$(origin $(v))), $(info $(v)=$($(v))))) 

## metric-obs : calculate the metric in observations
metric-obs : ${METRIC_OBS}
${METRIC_OBS} : 
	${FILEIO} ${OBS_DATA} $@ ${METRIC_OPTIONS} --metadata_file ${OBS_CONFIG}

# # metric-obs-analysis : analyse metric in observations
# metric-obs-analysis : obs_${REGION}.ipynb
# obs_${REGION_NAME}.ipynb : obs.ipynb ${METRIC_OBS} ${NINO_OBS}
# 	${PAPERMILL} -p obs_file $(word 2,$^) -p region_name ${REGION} -p nino_file $(word 3,$^) $< $@	

## metric-forecast : calculate metric in forecast ensemble
metric-forecast : ${METRIC_FCST}
${METRIC_FCST} : ${FCST_DATA}
	cp job.sh job_${MODEL}.sh
	echo "${FILEIO} $< $@ --forecast ${METRIC_OPTIONS} ${METRIC_OPTIONS_FCST} ${MODEL_IO_OPTIONS}" >> job_${MODEL}.sh
	qsub job_${MODEL}.sh

## independence-test : independence test for different lead times
independence-test : ${INDEPENDENCE_FILE}
${INDEPENDENCE_FILE} : ${METRIC_FCST}
	${INDEPENDENCE} $< ${VAR} $@ ${INDEPENDENCE_OPTIONS}

## independence-test : independence test for different lead times
independence-test-plot : ${INDEPENDENCE_PLOT}
${INDEPENDENCE_PLOT} : ${METRIC_FCST}
	${INDEPENDENCE} $< ${VAR} $@

# ## Stability test file (median) # todo
# stability-test-file-median : ${STABILITY_FILE_MEDIAN}
# ${STABILITY_FILE_MEDIAN} : ${METRIC_FCST}
# 	${STABILITY} $< ${VAR} ${METRIC} --outfile $@ 

## Stability test plot (empirical)
stability-test-empirical : ${STABILITY_PLOT_EMPIRICAL}
${STABILITY_PLOT_EMPIRICAL} : ${METRIC_FCST}
	${STABILITY} $< ${VAR} ${METRIC} --start_years ${STABILITY_START_YEARS} --outfile $@ --return_method empirical --units ${METRIC_PLOT_LABEL} --ylim 0 ${METRIC_PLOT_UPPER_LIMIT}
# --uncertainty

## Stability test plot (GEV fit)
stability-test-gev : ${STABILITY_PLOT_GEV}
${STABILITY_PLOT_GEV} : ${METRIC_FCST}
	${STABILITY} $< ${VAR} ${METRIC} --start_years ${STABILITY_START_YEARS} --outfile $@ --return_method gev --units ${METRIC_PLOT_LABEL} --ylim 0 ${METRIC_PLOT_UPPER_LIMIT}
# --uncertainty

## Additive bias-corrected model data file
bias-correction-additive : ${METRIC_FCST_ADDITIVE_BC}
${METRIC_FCST_ADDITIVE_BC} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} additive $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Multiplicative bias-corrected model data file
bias-correction-multiplicative : ${METRIC_FCST_MULTIPLICATIVE_BC}
${METRIC_FCST_MULTIPLICATIVE_BC} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} multiplicative $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Similarity test between observations and model data file
similarity-test : ${SIMILARITY_FILE}
${SIMILARITY_FILE} : ${METRIC_FCST} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Similarity test between observations and additive bias-corrected model data file
similarity-test-additive-bias : ${SIMILARITY_ADDITIVE_BC_FILE}
${SIMILARITY_ADDITIVE_BC_FILE} : ${METRIC_FCST_ADDITIVE_BC} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Similarity test between observations and multiplicative bias-corrected model data file
similarity-test-multiplicative-bias : ${SIMILARITY_MULTIPLICATIVE_BC_FILE}
${SIMILARITY_MULTIPLICATIVE_BC_FILE} : ${METRIC_FCST_MULTIPLICATIVE_BC} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Moments test between observations and model data file
moments-test : ${MOMENTS_PLOT}
${MOMENTS_PLOT} : ${METRIC_FCST} ${METRIC_OBS}
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units '${UNITS}'

## Moments test between observations and additive bias-corrected data file
moments-test-additive-bias : ${MOMENTS_ADDITIVE_BC_PLOT}
${MOMENTS_ADDITIVE_BC_PLOT} : ${METRIC_FCST} ${METRIC_OBS} ${METRIC_FCST_ADDITIVE_BC} 
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --bias_file $(word 3,$^) --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units '${UNITS}'

## Moments test between observations and multiplicative bias-corrected model data file
moments-test-multiplicative-bias : ${MOMENTS_MULTIPLICATIVE_BC_PLOT}
${MOMENTS_MULTIPLICATIVE_BC_PLOT} : ${METRIC_FCST} ${METRIC_OBS} ${METRIC_FCST_MULTIPLICATIVE_BC} 
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --bias_file $(word 3,$^) --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units '${UNITS}'

## Stationary GEV parameters for obs data
gev-params-stationary-obs : ${GEV_STATIONARY_OBS}
${GEV_STATIONARY_OBS} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} ${GEV_OBS_OPTIONS}

## Stationary GEV parameters for obs data (max event removed)
gev-params-stationary-obs-drop-max : ${GEV_STATIONARY_OBS_DROP_MAX}
${GEV_STATIONARY_OBS_DROP_MAX} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} ${GEV_OBS_OPTIONS} --drop_max

## Stationary GEV parameters for model data
gev-params-stationary : ${GEV_STATIONARY}
${GEV_STATIONARY} : ${METRIC_FCST}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Stationary GEV parameters for additive bias-corrected model dataa
gev-params-stationary-additive-bias : ${GEV_STATIONARY_ADDITIVE_BC}
${GEV_STATIONARY_ADDITIVE_BC} : ${METRIC_FCST_ADDITIVE_BC}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Stationary GEV parameters for multiplicative bias-corrected model data
gev-params-stationary-multiplicative-bias : ${GEV_STATIONARY_MULTIPLICATIVE_BC}
${GEV_STATIONARY_MULTIPLICATIVE_BC} : ${METRIC_FCST_MULTIPLICATIVE_BC}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for obs data
gev-params-nonstationary-obs : ${GEV_NONSTATIONARY_OBS}
${GEV_NONSTATIONARY_OBS} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} ${GEV_OBS_OPTIONS} 

## Non-stationary GEV parameters for obs data (max event removed)
gev-params-nonstationary-obs-drop-max : ${GEV_NONSTATIONARY_OBS_DROP_MAX}
${GEV_NONSTATIONARY_OBS_DROP_MAX} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} ${GEV_OBS_OPTIONS} --drop_max

## Non-stationary GEV parameters for model data
gev-params-nonstationary : ${GEV_NONSTATIONARY}
${GEV_NONSTATIONARY} : ${METRIC_FCST}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for additive bias-corrected model data
gev-params-nonstationary-additive-bias : ${GEV_NONSTATIONARY_ADDITIVE_BC}
${GEV_NONSTATIONARY_ADDITIVE_BC} : ${METRIC_FCST_ADDITIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for multiplicative bias-corrected model data
gev-params-nonstationary-multiplicative-bias : ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC}
${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} : ${METRIC_FCST_MULTIPLICATIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Best GEV parameters for obs data
gev-params-best-obs : ${GEV_BEST_OBS}
${GEV_BEST_OBS} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} ${GEV_OBS_OPTIONS} --pick_best_model ${GEV_TEST}

## Best GEV parameters for obs data (max event removed)
gev-params-best-obs-drop-max : ${GEV_BEST_OBS_DROP_MAX}
${GEV_BEST_OBS_DROP_MAX} : ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} ${GEV_OBS_OPTIONS} --drop_max --pick_best_model ${GEV_TEST}

## Best GEV parameters for model data
gev-params-best : ${GEV_BEST}
${GEV_BEST} : ${METRIC_FCST}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --pick_best_model ${GEV_TEST} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Best GEV parameters for additive bias-corrected model data
gev-params-best-additive-bias : ${GEV_BEST_ADDITIVE_BC}
${GEV_BEST_ADDITIVE_BC} : ${METRIC_FCST_ADDITIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --pick_best_model ${GEV_TEST} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Best GEV parameters for multiplicative bias-corrected model data
gev-params-best-multiplicative-bias : ${GEV_BEST_MULTIPLICATIVE_BC}
${GEV_BEST_MULTIPLICATIVE_BC} : ${METRIC_FCST_MULTIPLICATIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --stack_dims --pick_best_model ${GEV_TEST} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}


## Combined targets
moments : ${MOMENTS_ADDITIVE_BC_PLOT} ${MOMENTS_MULTIPLICATIVE_BC_PLOT} ${MOMENTS_RAW_PLOT}
forecast-diagnostics : ${INDEPENDENCE_FILE} ${METRIC_FCST_ADDITIVE_BC} ${METRIC_FCST_MULTIPLICATIVE_BC} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC_FILE} ${SIMILARITY_MULTIPLICATIVE_BC_FILE}
forecast-gev-params : ${GEV_STATIONARY} ${GEV_NONSTATIONARY} ${GEV_BEST} 
forecast-gev-params-additive-bc : ${GEV_STATIONARY_ADDITIVE_BC} ${GEV_NONSTATIONARY_ADDITIVE_BC} ${GEV_BEST_ADDITIVE_BC}
forecast-gev-params-multiplicative-bc : ${GEV_STATIONARY_MULTIPLICATIVE_BC} ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} ${GEV_BEST_MULTIPLICATIVE_BC}
obs-gev-params : ${GEV_NONSTATIONARY_OBS} ${GEV_NONSTATIONARY_OBS_DROP_MAX} ${GEV_STATIONARY_OBS} ${GEV_STATIONARY_OBS_DROP_MAX} ${GEV_BEST_OBS} ${GEV_BEST_OBS_DROP_MAX}

## Analysis notebook of the metric from the model data
metric-forecast-analysis : ${NOTEBOOK_OUT_DIR}/analysis_${MODEL}.ipynb
${NOTEBOOK_OUT_DIR}/analysis_${MODEL}.ipynb : ${NOTEBOOK_IN_DIR}/analysis.ipynb ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BC} ${METRIC_FCST_MULTIPLICATIVE_BC} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC} ${SIMILARITY_MULTIPLICATIVE_BC} ${INDEPENDENCE_FILE} ${INDEPENDENCE_PLOT} ${STABILITY_PLOT_EMPIRICAL} ${STABILITY_PLOT_GEV} ${MOMENTS_ADDITIVE_BC_PLOT} ${MOMENTS_MULTIPLICATIVE_BC_PLOT} ${MOMENTS_PLOT} ${FCST_DATA}
	${PAPERMILL} -p model_name ${MODEL} -p metric ${METRIC} -p var ${VAR} -p metric_plot_label ${METRIC_PLOT_LABEL} -p metric_plot_upper_limit ${METRIC_PLOT_UPPER_LIMIT} -p obs_file $(word 2,$^) -p model_file $(word 3,$^) -p model_add_bc_file $(word 4,$^) -p model_mulc_bc_file $(word 5,$^) -p similarity_raw_file $(word 6,$^)-p similarity_add_bc_file $(word 7,$^) -p similarity_mulc_bc_file $(word 8,$^) -p independence_file $(word 9,$^) -p independence_plot $(word 10,$^) -p stability_plot_empirical $(word 11,$^) -p stability_plot_gev $(word 12,$^) -p moments_add_plot $(word 13,$^) -p moments_mulc_plot $(word 14,$^) -p moments_raw_plot $(word 15,$^) -p min_lead ${MIN_LEAD} -p region_name ${REGION} -p shape_file ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP} -p file_list $(word 16,$^) $< $@

## Spatial analysis notebook of the metric from the model data
# Notes: 
# - If plot_additive_bc or plot_multiplicative_bc are True, then the corresponding bias-corrected files are required. 
# - Only the gev_params_nonstationary files are required. The optional "stationary" and "best" diagnostic plots will be plotted if the files exist.
# ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BC} ${METRIC_FCST_MULTIPLICATIVE_BC} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC} ${SIMILARITY_MULTIPLICATIVE_BC} ${INDEPENDENCE_FILE} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC_FILE} ${SIMILARITY_MULTIPLICATIVE_BC_FILE} ${GEV_NONSTATIONARY} ${GEV_NONSTATIONARY_ADDITIVE_BC} ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} ${GEV_BEST} ${GEV_BEST_ADDITIVE_BC} ${GEV_BEST_MULTIPLICATIVE_BC}
metric-forecast-spatial-analysis : ${NOTEBOOK_OUT_DIR}/spatial_analysis_${METRIC}_${MODEL}.ipynb 
${NOTEBOOK_OUT_DIR}/spatial_analysis_${METRIC}_${MODEL}.ipynb : ${NOTEBOOK_IN_DIR}/spatial_analysis.ipynb 
	${PAPERMILL} $< $@ -p model_name ${MODEL} -p metric ${METRIC} -p var ${VAR} -p obs_name ${OBS_LABEL} -p reference_time_period '${REFERENCE_TIME_PERIOD}' -p time_agg ${TIME_AGG} -p covariate_base ${COVARIATE_BASE} -p gev_trend_period ${GEV_TREND_PERIOD} -p plot_dict ${PLOT_DICT} -p fig_dir ${FIG_DIR} -p plot_additive_bc ${PLOT_ADDITIVE_BC} -p plot_multiplicative_bc ${PLOT_MULTIPLICATIVE_BC} -p shapefile ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP} -p obs_file ${METRIC_OBS} -p model_file ${METRIC_FCST} -p model_add_bc_file ${METRIC_FCST_ADDITIVE_BC} -p model_mulc_bc_file ${METRIC_FCST_MULTIPLICATIVE_BC} -p independence_file ${INDEPENDENCE_FILE} -p independence_plot ${INDEPENDENCE_PLOT} -p min_lead_spatial_agg ${MIN_LEAD_SHAPE_SPATIAL_AGG} -p similarity_raw_file ${SIMILARITY_FILE} -p similarity_add_bc_file ${SIMILARITY_ADDITIVE_BC_FILE} -p similarity_mulc_bc_file ${SIMILARITY_MULTIPLICATIVE_BC_FILE} -p similarity_raw_plot ${SIMILARITY_PLOT} -p similarity_add_bc_plot ${SIMILARITY_ADDITIVE_BC_PLOT} -p similarity_mulc_bc_plot ${SIMILARITY_MULTIPLICATIVE_BC_PLOT} -p gev_params_nonstationary_file ${GEV_NONSTATIONARY} -p gev_params_nonstationary_add_bc_file ${GEV_NONSTATIONARY_ADDITIVE_BC} -p gev_params_nonstationary_mulc_bc_file ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} -p gev_params_stationary_file ${GEV_STATIONARY} -p gev_params_stationary_add_bc_file ${GEV_STATIONARY_ADDITIVE_BC} -p gev_params_stationary_mulc_bc_file ${GEV_STATIONARY_MULTIPLICATIVE_BC} -p gev_params_best_file ${GEV_BEST} -p gev_params_best_add_bc_file ${GEV_BEST_ADDITIVE_BC} -p gev_params_best_mulc_bc_file ${GEV_BEST_MULTIPLICATIVE_BC}

metric-obs-spatial-analysis : ${NOTEBOOK_OUT_DIR}/spatial_analysis_${METRIC}_${OBS_DATASET}.ipynb ${METRIC_OBS} ${GEV_NONSTATIONARY_OBS} ${GEV_NONSTATIONARY_OBS_DROP_MAX}
${NOTEBOOK_OUT_DIR}/spatial_analysis_${METRIC}_${OBS_DATASET}.ipynb : ${NOTEBOOK_IN_DIR}/spatial_obs.ipynb
	${PAPERMILL} $< $@ -p obs_name ${OBS_LABEL} -p metric ${METRIC} -p var ${VAR} -p reference_time_period '${REFERENCE_TIME_PERIOD}' -p time_agg ${TIME_AGG} -p covariate_base ${COVARIATE_BASE} -p gev_trend_period ${GEV_TREND_PERIOD} -p plot_dict ${PLOT_DICT} -p fig_dir ${FIG_DIR} -p obs_file ${METRIC_OBS} -p gev_params_nonstationary_file ${GEV_NONSTATIONARY_OBS} -p gev_params_nonstationary_drop_max_file ${GEV_NONSTATIONARY_OBS_DROP_MAX} -p gev_params_best_file ${GEV_BEST_OBS} -p gev_params_stationary_file ${GEV_STATIONARY_OBS} -p shapefile ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP}

.PHONY: help moments

## help : show this message
help :
	@echo 'make [target] [-Bnf] PROJECT_DETAILS=project.mk MODEL_DETAILS=model.mk OBS_DETAILS=obs.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'
