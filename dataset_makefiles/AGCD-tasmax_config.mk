# Configuration file for Makefile workflows

OBS_DATASET=AGCD-CSIRO
OBS_DATA := $(sort $(wildcard /g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005_*_daily.nc))
OBS_CONFIG=/g/data/xv83/unseen-projects/code/dataset_config/dataset_agcd_daily.yml
OBS_TIME_PERIOD=1901-2023