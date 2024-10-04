.PHONY: help moments

include ${PROJECT_DETAILS}
include ${MODEL_DETAILS}
include ${OBS_DETAILS}

DASK_CONFIG=/g/data/xv83/unseen-projects/code/dask_local.yml
ENV_DIR=/g/data/xv83/as3189/conda/envs/unseen
FCST_DATA=/g/data/xv83/unseen-projects/code/file_lists/${MODEL}_${EXPERIMENT}_${VAR}_files.txt

FILEIO=${ENV_DIR}/bin/fileio
PAPERMILL=${ENV_DIR}/bin/papermill
INDEPENDENCE=${ENV_DIR}/bin/independence
BIAS_CORRECTION=${ENV_DIR}/bin/bias_correction
SIMILARITY=${ENV_DIR}/bin/similarity

METRIC_OBS=${PROJECT_DIR}/data/${METRIC}_${OBS_DATASET}_${OBS_TIME_PERIOD}_${TIMESCALE}_${REGION}.nc
METRIC_FCST=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
INDEPENDENCE_FILE=${PROJECT_DIR}/data/independence-test_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
INDEPENDENCE_PLOT=${PROJECT_DIR}/figures/${METRIC}/independence-test_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
METRIC_FCST_ADDITIVE_BIAS_CORRECTED=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
SIMILARITY_ADDITIVE_BIAS=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
SIMILARITY_MULTIPLICATIVE_BIAS=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
SIMILARITY_RAW=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.nc

## metric-obs : calculate the metric in observations
metric-obs : ${METRIC_OBS}
${METRIC_OBS} :
	${FILEIO} ${OBS_DATA} $@ ${METRIC_OPTIONS} --metadata_file ${OBS_CONFIG}

## metric-obs-analysis : analyse metric in observations
#metric-obs-analysis : AGCD_${REGION}.ipynb
#obs_${REGION_NAME}.ipynb : obs.ipynb ${METRIC_OBS}
#	${PAPERMILL} -p obs_file $(word 2,$^) -p region_name ${REGION} -p nino_file $(word 3,$^) $< $@	

## metric-forecast : calculate metric in forecast ensemble
metric-forecast : ${METRIC_FCST}
${METRIC_FCST} : ${FCST_DATA}
	cp job.sh job_${MODEL}.sh
	echo "${FILEIO} $< $@ --forecast ${METRIC_OPTIONS} ${METRIC_OPTIONS_FCST} ${MODEL_IO_OPTIONS}" >> job_${MODEL}.sh
	@if [ ! -f $@ ]; then \
		qsub job_${MODEL}.sh; \
	fi

## independence-test : independence test for different lead times
independence-test : ${INDEPENDENCE_FILE}
${INDEPENDENCE_FILE} : ${METRIC_FCST}
	cp job_independence.sh job_independence_${MODEL}.sh
	echo "${INDEPENDENCE} $< ${VAR} $@ --plot_outfile ${INDEPENDENCE_PLOT} ${INDEPENDENCE_OPTIONS}" >> job_independence_${MODEL}.sh
	@if [ ! -f $@ ]; then \
		qsub job_independence_${MODEL}.sh; \
	fi
	
## bias-correction-additive : additive bias corrected forecast data using observations
bias-correction-additive : ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED}
${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} additive $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${INDEPENDENCE_FILE} ${MIN_LEAD_OPTIONS}

## bias-correction-multiplicative : multiplicative bias corrected forecast data using observations
bias-correction-multiplicative : ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED}
${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} multiplicative $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${INDEPENDENCE_FILE} ${MIN_LEAD_OPTIONS}

## similarity-test-additive-bias : similarity test between observations and additive bias corrected forecast
similarity-test-additive-bias : ${SIMILARITY_ADDITIVE_BIAS}
${SIMILARITY_ADDITIVE_BIAS} : ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_LEAD_OPTIONS}

## similarity-test-multiplicative-bias : similarity test between observations and multiplicative bias corrected forecast
similarity-test-multiplicative-bias : ${SIMILARITY_MULTIPLICATIVE_BIAS}
${SIMILARITY_MULTIPLICATIVE_BIAS} : ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_LEAD_OPTIONS}

## similarity-test-raw : similarity test between observations and raw forecast
similarity-test-raw : ${SIMILARITY_RAW}
${SIMILARITY_RAW} : ${METRIC_FCST} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${INDEPENDENCE_FILE} ${MIN_LEAD_OPTIONS}

## metric-f0000orecast-analysis : analysis of the metric from forecast data
metric-forecast-analysis : analysis_${MODEL}.ipynb
analysis_${MODEL}.ipynb : analysis.ipynb ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} ${SIMILARITY_ADDITIVE_BIAS} ${SIMILARITY_MULTIPLICATIVE_BIAS} ${SIMILARITY_RAW} ${INDEPENDENCE_PLOT}  ${FCST_DATA}
	${PAPERMILL} -p metric ${METRIC} -p var ${VAR} -p metric_plot_label ${METRIC_PLOT_LABEL} -p metric_plot_upper_limit ${METRIC_PLOT_UPPER_LIMIT} -p obs_file $(word 2,$^) -p model_file $(word 3,$^) -p model_add_bc_file $(word 4,$^) -p model_mulc_bc_file $(word 5,$^) -p similarity_add_bc_file $(word 6,$^) -p similarity_mulc_bc_file $(word 7,$^) -p similarity_raw_file $(word 8,$^) -p independence_file $(word 9,$^)-p independence_plot $(word 9,$^) -p region_name ${REGION} -p shape_file ${SHAPEFILE} -p shape_overlap ${SHAPE_OVERLAP} -p file_list $(word 15,$^) $< $@

## help : show this message
help :
	@echo 'make [target] [-Bnf] PROJECT_DETAILS=project.mk MODEL_DETAILS=model.mk OBS_DETAILS=obs.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'
