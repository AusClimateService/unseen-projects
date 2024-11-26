include ${PROJECT_DETAILS}
include ${MODEL_DETAILS}
include ${OBS_DETAILS}

DASK_CONFIG=/g/data/xv83/unseen-projects/code/dask_local.yml
ENV_DIR?=/g/data/xv83/unseen-projects/unseen_venv

# Optional arguments
FIG_DIR?=${PROJECT_DIR}/figures
GEV_BEST_MODEL_TEST?=bic
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
GEV_STATIONARY=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
GEV_STATIONARY_ADDITIVE_BC=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
GEV_STATIONARY_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/gev_params_stationary_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
GEV_NONSTATIONARY=${PROJECT_DIR}/data/gev_params_nonstationary_bic_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
GEV_NONSTATIONARY_ADDITIVE_BC=${PROJECT_DIR}/data/gev_params_nonstationary_${GEV_BEST_MODEL_TEST}_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
GEV_NONSTATIONARY_MULTIPLICATIVE_BC=${PROJECT_DIR}/data/gev_params_nonstationary_bic_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
NINO_FCST=${PROJECT_DIR}/data/nino34-anomaly_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_base-${BASE_PERIOD_TEXT}.nc
NINO_OBS=${PROJECT_DIR}/data/nino34-anomaly_HadISST_1870-2022_base-1981-2010.nc

# ## metric-obs : calculate the metric in observations
# metric-obs : ${METRIC_OBS}
# ${METRIC_OBS} :
# 	@if [ ! -f $@ ]; then \
# 		${FILEIO} ${OBS_DATA} $@ ${METRIC_OPTIONS} --metadata_file ${OBS_CONFIG}; \
# 	fi

# # metric-obs-analysis : analyse metric in observations
# metric-obs-analysis : obs_${REGION}.ipynb
# obs_${REGION_NAME}.ipynb : obs.ipynb ${METRIC_OBS} ${NINO_OBS}
# 	${PAPERMILL} -p obs_file $(word 2,$^) -p region_name ${REGION} -p nino_file $(word 3,$^) $< $@	

## metric-forecast : calculate metric in forecast ensemble
metric-forecast : ${METRIC_FCST}
${METRIC_FCST} : ${FCST_DATA}
	@if [ ! -f $@ ]; then \
		cp job.sh job_${MODEL}.sh
		echo "${FILEIO} $< $@ --forecast ${METRIC_OPTIONS} ${METRIC_OPTIONS_FCST} ${FCST_IO_OPTIONS}" >> job_${MODEL}.sh
		qsub job_${MODEL}.sh; \
	fi

## independence-test : independence test for different lead times # todo: Update analysis for file
independence-test : ${INDEPENDENCE_FILE}
${INDEPENDENCE_FILE} : ${METRIC_FCST}
	@if [ ! -f $@ ]; then \
		cp job_independence.sh job_independence_${MODEL}.sh
		echo "${INDEPENDENCE} $< ${VAR} $@ ${INDEPENDENCE_OPTIONS}" >> job_independence_${MODEL}.sh
		qsub job_independence_${MODEL}.sh; \
	fi
# ## independence-test : independence test for different lead times
# independence-test : ${INDEPENDENCE_PLOT}
# ${INDEPENDENCE_PLOT} : ${METRIC_FCST}
# 	${INDEPENDENCE} $< ${VAR} $@

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
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units ${UNITS}

## Moments test between observations and additive bias-corrected data file
moments-test-additive-bias : ${MOMENTS_ADDITIVE_BC_PLOT}
${MOMENTS_ADDITIVE_BC_PLOT} : ${METRIC_FCST} ${METRIC_OBS} ${METRIC_FCST_ADDITIVE_BC} 
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --bias_file $(word 3,$^) --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units ${UNITS}

## Moments test between observations and multiplicative bias-corrected model data file
moments-test-multiplicative-bias : ${MOMENTS_MULTIPLICATIVE_BC_PLOT}
${MOMENTS_MULTIPLICATIVE_BC_PLOT} : ${METRIC_FCST} ${METRIC_OBS} ${METRIC_FCST_MULTIPLICATIVE_BC} 
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --bias_file $(word 3,$^) --min_lead ${MIN_LEAD} ${MIN_IND_LEAD_OPTIONS} --units ${UNITS}

## Stationary GEV parameters for model data
gev-params-stationary : ${GEV_STATIONARY}
${GEV_STATIONARY} : ${METRIC_FCST}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Stationary GEV parameters for additive bias-corrected model dataa
gev-params-stationary-additive-bias : ${GEV_STATIONARY_ADDITIVE_BC}
${GEV_STATIONARY_ADDITIVE_BC} : ${METRIC_FCST_ADDITIVE_BC}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Stationary GEV parameters for multiplicative bias-corrected model data
gev-params-stationary-multiplicative-bias : ${GEV_STATIONARY_MULTIPLICATIVE_BC}
${GEV_STATIONARY_MULTIPLICATIVE_BC} : ${METRIC_FCST_MULTIPLICATIVE_BC}
	${EVA} $< ${VAR} $@ --stationary ${GEV_STATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for model data
gev-params-nonstationary : ${GEV_NONSTATIONARY}
${GEV_NONSTATIONARY} : ${METRIC_FCST} ${METRIC_OBS}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for additive bias-corrected model data
gev-params-nonstationary-additive-bias : ${GEV_NONSTATIONARY_ADDITIVE_BC}
${GEV_NONSTATIONARY_ADDITIVE_BC} : ${METRIC_FCST_ADDITIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

## Non-stationary GEV parameters for multiplicative bias-corrected model data
gev-params-nonstationary-multiplicative-bias : ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC}
${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} : ${METRIC_FCST_MULTIPLICATIVE_BC}
	${EVA} $< ${VAR} $@ --nonstationary ${GEV_NONSTATIONARY_OPTIONS} --min_lead ${INDEPENDENCE_FILE} ${MIN_IND_LEAD_OPTIONS}

gev-params : gev-params-stationary gev-params-nonstationary gev-params-stationary-additive-bias gev-params-nonstationary-additive-bias gev-params-stationary-multiplicative-bias gev-params-nonstationary-multiplicative-bias

## Analysis notebook of the metric from the model data
metric-forecast-analysis : analysis_${MODEL}.ipynb
analysis_${MODEL}.ipynb : analysis.ipynb ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BC} ${METRIC_FCST_MULTIPLICATIVE_BC} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC} ${SIMILARITY_MULTIPLICATIVE_BC} ${INDEPENDENCE_FILE} ${INDEPENDENCE_PLOT} ${STABILITY_PLOT_EMPIRICAL} ${STABILITY_PLOT_GEV} ${MOMENTS_ADDITIVE_BC_PLOT} ${MOMENTS_MULTIPLICATIVE_BC_PLOT} ${MOMENTS_PLOT} ${FCST_DATA}
	${PAPERMILL} -p model_name ${MODEL} -p metric ${METRIC} -p var ${VAR} -p metric_plot_label ${METRIC_PLOT_LABEL} -p metric_plot_upper_limit ${METRIC_PLOT_UPPER_LIMIT} -p obs_file $(word 2,$^) -p model_file $(word 3,$^) -p model_add_bc_file $(word 4,$^) -p model_mulc_bc_file $(word 5,$^) -p similarity_raw_file $(word 6,$^)-p similarity_add_bc_file $(word 7,$^) -p similarity_mulc_bc_file $(word 8,$^) -p independence_file $(word 9,$^) -p independence_plot $(word 10,$^) -p stability_plot_empirical $(word 11,$^) -p stability_plot_gev $(word 12,$^) -p moments_add_plot $(word 13,$^) -p moments_mulc_plot $(word 14,$^) -p moments_raw_plot $(word 15,$^) -p min_lead ${MIN_LEAD} -p region_name ${REGION} -p shape_file ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP} -p file_list $(word 16,$^) $< $@

## Spatial analysis notebook of the metric from the model data
## prereq files (todo: move back): ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BC} ${METRIC_FCST_MULTIPLICATIVE_BC} ${INDEPENDENCE_FILE} ${SIMILARITY_FILE} ${SIMILARITY_ADDITIVE_BC_FILE} ${SIMILARITY_MULTIPLICATIVE_BC_FILE} ${GEV_STATIONARY} ${GEV_NONSTATIONARY} ${GEV_STATIONARY_ADDITIVE_BC} ${GEV_NONSTATIONARY_ADDITIVE_BC} ${GEV_STATIONARY_MULTIPLICATIVE_BC} ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC} ${FCST_DATA} 
metric-forecast-spatial-analysis : spatial_analysis_${METRIC}_${MODEL}.ipynb
spatial_analysis_${METRIC}_${MODEL}.ipynb : spatial_analysis.ipynb
	${PAPERMILL} $< $@ -p model_name ${MODEL} -p metric ${METRIC} -p var ${VAR} -p time_freq ${TIME_FREQ} -p time_agg ${TIME_AGG} -p covariate_base ${GEV_COVARIATE_BASE} -p gev_trend_period ${GEV_TREND_PERIOD} -p plot_dict ${PLOT_DICT} -p fig_dir ${FIG_DIR} -p bias_correction ${BIAS_CORRECTION} -p plot_additive_bc ${PLOT_ADDITIVE_BC} -p plot_multiplicative_bc ${PLOT_MULTIPLICATIVE_BC} -p shapefile ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP} -p obs_file ${METRIC_OBS} -p model_file ${METRIC_FCST} -p model_add_bc_file ${METRIC_FCST_ADDITIVE_BC} -p model_mulc_bc_file ${METRIC_FCST_MULTIPLICATIVE_BC} -p independence_file ${INDEPENDENCE_FILE} -p independence_plot ${INDEPENDENCE_PLOT} -p min_lead_spatial_agg ${MIN_LEAD_SHAPE_SPATIAL_AGG} -p similarity_raw_file ${SIMILARITY_FILE} -p similarity_add_bc_file ${SIMILARITY_ADDITIVE_BC_FILE} -p similarity_mulc_bc_file ${SIMILARITY_MULTIPLICATIVE_BC_FILE} -p similarity_raw_plot ${SIMILARITY_PLOT} -p similarity_add_bc_plot ${SIMILARITY_ADDITIVE_BC_PLOT} -p similarity_mulc_bc_plot ${SIMILARITY_MULTIPLICATIVE_BC_PLOT} -p gev_params_stationary_file ${GEV_STATIONARY} -p gev_params_nonstationary_file ${GEV_NONSTATIONARY} -p gev_params_stationary_add_bc_file ${GEV_STATIONARY_ADDITIVE_BC} -p gev_params_nonstationary_add_bc_file ${GEV_NONSTATIONARY_ADDITIVE_BC} -p gev_params_stationary_mulc_bc_file ${GEV_STATIONARY_MULTIPLICATIVE_BC} -p gev_params_nonstationary_mulc_bc_file ${GEV_NONSTATIONARY_MULTIPLICATIVE_BC}

metric-obs-spatial-analysis : spatial_analysis_${METRIC}_obs_${MODEL}.ipynb
spatial_analysis_${METRIC}_obs_${MODEL}.ipynb : spatial_obs.ipynb
	${PAPERMILL} $< $@ -p model_name ${MODEL} -p metric ${METRIC} -p var ${VAR} -p region_name ${REGION} -p time_freq ${TIME_FREQ} -p fig_dir ${FIG_DIR}-p obs_file ${METRIC_OBS} -p gev_params_stationary_file ${GEV_STATIONARY} -p gev_params_nonstationary_file ${GEV_NONSTATIONARY} -p shapefile ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP}

.PHONY: help moments
moments : ${MOMENTS_ADDITIVE_BC_PLOT} ${MOMENTS_MULTIPLICATIVE_BC_PLOT} ${MOMENTS_RAW_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] PROJECT_DETAILS=project.mk MODEL_DETAILS=model.mk OBS_DETAILS=obs.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'
