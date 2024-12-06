# Configuration file for Makefile workflows

OBS_DATASET=AGCD-CSIRO_r05
# Optional: set the name of the dataset for spatial plot labels
OBS_LABEL=AGCD
# OBS_DATA := $(sort $(wildcard /g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005_*_daily.nc))
OBS_DATA := /g/data/xv83/unseen-projects/outputs/hazards/data/tmax_AGCD-CSIRO_r05_1901-2024.nc
OBS_CONFIG=/g/data/xv83/unseen-projects/code/dataset_config/dataset_agcd_daily.yml
OBS_TIME_PERIOD=1901-2024