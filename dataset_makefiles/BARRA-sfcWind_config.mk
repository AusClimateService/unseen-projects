# Configuration file for Makefile workflows

OBS_DATASET=BARRA-R2
OBS_DATA := $(sort $(wildcard /g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/sfcWind/latest/*.nc))
OBS_CONFIG=/home/599/dbi599/unseen-projects/dataset_config/dataset_barra.yml
OBS_TIME_PERIOD=1979-2025
