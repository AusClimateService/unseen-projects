# Configuration file for Makefile workflows

MODEL=MPI-ESM1-2-HR
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1970-01-01 2018-12-31
BASE_PERIOD_TEXT=1970-2018
TIME_PERIOD_TEXT=196011-201811
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 10 --metadata_file /g/data/xv83/unseen-projects/code/dataset_config/dataset_dcpp.yml
