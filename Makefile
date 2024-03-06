.PHONY: help moments

include ${PROJECT_CONFIG}
include ${MODEL_CONFIG}

DASK_CONFIG=dask_local.yml

FCST_DATA=/g/data/xv83/unseen-projects/code/file_lists/${MODEL}_${EXPERIMENT}_${VAR}_files.txt
OBS_DATA := $(sort $(wildcard /g/data/xv83/agcd-csiro/precip/daily/precip-total_AGCD-CSIRO_r005_*_daily.nc))
OBS_CONFIG=/g/data/xv83/unseen-projects/code/dataset_config/dataset_agcd_daily.yml

METRIC_OBS=${PROJECT_DIR}/data/${METRIC}_${OBS_DATASET}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION_NAME}.nc
METRIC_FCST=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.nc
INDEPENDENCE_PLOT=${PROJECT_DIR}/figures/independence-test_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
STABILITY_PLOT_EMPIRICAL=${PROJECT_DIR}/figures/stability-test-empirical_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
STABILITY_PLOT_GEV=${PROJECT_DIR}/figures/stability-test-gev_${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}.png
RX15DAY_FCST_ADDITIVE_BIAS_CORRECTED=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
RX15DAY_FCST_MULTIPLICATIVE_BIAS_CORRECTED=${PROJECT_DIR}/data/${METRIC}_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
SIMILARITY_ADDITIVE_BIAS=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.nc
SIMILARITY_MULTIPLICATIVE_BIAS=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.nc
SIMILARITY_RAW=${PROJECT_DIR}/data/similarity-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.nc
MOMENTS_ADDITIVE_BIAS_PLOT=${PROJECT_DIR}/figures/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-additive.png
MOMENTS_MULTIPLICATIVE_BIAS_PLOT=${PROJECT_DIR}/figures/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_bias-corrected-${OBS_DATASET}-multiplicative.png
MOMENTS_RAW_PLOT=${PROJECT_DIR}/figures/moments-test_${METRIC}_${MODEL}-${EXPERIMENT}_${BASE_PERIOD_TEXT}_${TIMESCALE}_${REGION}_${OBS_DATASET}.png
NINO_FCST=${PROJECT_DIR}/data/nino34-anomaly_${MODEL}-${EXPERIMENT}_${TIME_PERIOD_TEXT}_base-${BASE_PERIOD_TEXT}.nc
NINO_OBS=${PROJECT_DIR}/data/nino34-anomaly_HadISST_1870-2022_base-1981-2010.nc

FILEIO=/g/data/xv83/dbi599/miniconda3/envs/unseen-processing/bin/fileio
PAPERMILL=/g/data/xv83/dbi599/miniconda3/envs/unseen2/bin/papermill
INDEPENDENCE=/g/data/xv83/dbi599/miniconda3/envs/unseen-processing/bin/independence
STABILITY=/g/data/xv83/dbi599/miniconda3/envs/unseen2/bin/stability
BIAS_CORRECTION=/g/data/xv83/dbi599/miniconda3/envs/unseen-processing/bin/bias_correction
SIMILARITY=/g/data/xv83/dbi599/miniconda3/envs/unseen-processing/bin/similarity
MOMENTS=/g/data/xv83/dbi599/miniconda3/envs/unseen-processing/bin/moments

## metric-obs : calculate the metric in observations
metric-obs : ${METRIC_OBS}
${RX15DAY_OBS} :
	${FILEIO} ${OBS_DATA} $@ ${METRIC_OPTIONS} --metadata_file ${OBS_CONFIG}

## metric-obs-analysis : analyse metric in observations
rx15day-obs-analysis : AGCD_${REGION}.ipynb
AGCD_${REGION_NAME}.ipynb : AGCD.ipynb ${METRIC_OBS} ${NINO_OBS}
	${PAPERMILL} -p rx15day_file $(word 2,$^) -p region_name ${REGION} -p nino_file $(word 3,$^) $< $@	

## metric-forecast : calculate metric in forecast ensemble
metric-forecast : ${METRIC_FCST}
${METRIC_FCST} : ${FCST_DATA}
	${FILEIO} $< $@ --forecast ${METRIC_OPTIONS} ${METRIC_OPTIONS_FCST} ${MODEL_IO_OPTIONS} --dask_config ${DASK_CONFIG}

## independence-test : independence test for different lead times
independence-test : ${INDEPENDENCE_PLOT}
${INDEPENDENCE_PLOT} : ${RX15DAY_FCST}
	${INDEPENDENCE} $< ${VAR} $@

## stability-test-empirical : stability tests (empirical)
stability-test-empirical : ${STABILITY_PLOT_EMPIRICAL}
${STABILITY_PLOT_EMPIRICAL} : ${METRIC_FCST}
	${STABILITY} $< ${VAR} ${METRIC} --start_years ${STABILITY_START_YEARS} --outfile $@ --return_method empirical --uncertainty --units "Rx15day (mm)" --ymax 550

## stability-test-gev : stability tests (GEV fit)
stability-test-gev : ${STABILITY_PLOT_GEV}
${STABILITY_PLOT_GEV} : ${RX15DAY_FCST}
	${STABILITY} $< ${VAR} ${METRIC} --start_years ${STABILITY_START_YEARS} --outfile $@ --return_method gev --uncertainty --units "Rx15day (mm)" --ymax 550

## bias-correction-additive : additive bias corrected forecast data using observations
bias-correction : ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED}
${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} additive $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${MIN_LEAD}

## bias-correction-multiplicative : multiplicative bias corrected forecast data using observations
bias-correction : ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED}
${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} : ${METRIC_FCST} ${METRIC_OBS}
	${BIAS_CORRECTION} $< $(word 2,$^) ${VAR} multiplicative $@ --base_period ${BASE_PERIOD} --rounding_freq A --min_lead ${MIN_LEAD}

## similarity-test-additive-bias : similarity test between observations and additive bias corrected forecast
similarity-test-additive-bias : ${SIMILARITY_ADDITIVE_BIAS}
${SIMILARITY_ADDITIVE_BIAS} : ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${MIN_LEAD}

## similarity-test-multiplicative-bias : similarity test between observations and multiplicative bias corrected forecast
similarity-test-multiplicative-bias : ${SIMILARITY_MULTIPLICATIVE_BIAS}
${SIMILARITY_MULTIPLICATIVE_BIAS} : ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${MIN_LEAD}

## similarity-test-raw : similarity test between observations and raw forecast
similarity-test-raw : ${SIMILARITY_RAW}
${SIMILARITY_RAW} : ${METRIC_FCST} ${METRIC_OBS}
	${SIMILARITY} $< $(word 2,$^) ${VAR} $@ --reference_time_period ${BASE_PERIOD} --min_lead ${MIN_LEAD}

## moments-test-additive-bias : moments test between observations and additive bias corrected forecast
moments-test-additive-bias : ${MOMENTS_ADDITIVE_BIAS_PLOT}
${MOMENTS_ADDITIVE_BIAS_PLOT} : ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} ${METRIC_OBS}
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --min_lead ${MIN_LEAD} --units mm

## moments-test-multiplicative-bias : moments test between observations and multiplicative bias corrected forecast
moments-test-multiplicative-bias : ${MOMENTS_MULTIPLICATIVE_BIAS_PLOT}
${MOMENTS_MULTIPLICATIVE_BIAS_PLOT} : ${METRIC_FCST} ${METRIC_OBS} ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} 
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --bias_file $(word 3,$^) --min_lead ${MIN_LEAD} --units mm

## moments-test-raw : moments test between observations and raw forecast
moments-test-raw : ${MOMENTS_RAW_PLOT}
${MOMENTS_RAW_PLOT} : ${METRIC_FCST} ${METRIC_OBS}
	${MOMENTS} $< $(word 2,$^) ${VAR} --outfile $@ --min_lead ${MIN_LEAD} --units mm

## metric-forecast-analysis : analysis of the metric from forecast data
metric-forecast-analysis : analysis_${MODEL}.ipynb
analysis_${MODEL}.ipynb : analysis.ipynb ${METRIC_OBS} ${METRIC_FCST} ${METRIC_FCST_ADDITIVE_BIAS_CORRECTED} ${METRIC_FCST_MULTIPLICATIVE_BIAS_CORRECTED} ${SIMILARITY_ADDITIVE_BIAS} ${SIMILARITY_MULTIPLICATIVE_BIAS} ${SIMILARITY_RAW} ${INDEPENDENCE_PLOT} ${STABILITY_PLOT} ${FCST_DATA}
	${PAPERMILL} -p metric ${METRIC} -p var ${VAR} -p obs_file $(word 2,$^) -p model_file $(word 3,$^) -p model_add_bc_file $(word 4,$^) -p model_mulc_bc_file $(word 5,$^) -p similarity_add_bc_file $(word 6,$^) -p similarity_mulc_bc_file $(word 7,$^) -p similarity_raw_file $(word 8,$^) -p independence_plot $(word 9,$^) -p stability_plot $(word 10,$^) -p model_name ${MODEL} -p min_lead ${MIN_LEAD} -p region_name ${REGION_NAME} -p shape_file ${SHAPEFILE} -p file_list $(word 11,$^) $< $@

moments : ${MOMENTS_ADDITIVE_BIAS_PLOT} ${MOMENTS_MULTIPLICATIVE_BIAS_PLOT} ${MOMENTS_RAW_PLOT}

## help : show this message
help :
	@echo 'make [target] [-Bnf] CONFIG=config_file.mk'
	@echo ''
	@echo 'valid targets:'
	@grep -h -E '^##' ${MAKEFILE_LIST} | sed -e 's/## //g' | column -t -s ':'
