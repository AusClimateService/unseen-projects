# Configuration file for Makefile workflows

MODEL=NorCPM1
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1970-01-01 2018-12-31
BASE_PERIOD_TEXT=1970-2018
TIME_PERIOD_TEXT=196010-201810
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 20 --metadata_file /g/data/xv83/unseen-projects/code/dataset_config/dataset_dcpp.yml
MODEL_NINO_OPTIONS=${MODEL_IO_OPTIONS} --lon_bnds 190 240 --lat_dim latitude --lon_dim longitude --agg_y_dim j --agg_x_dim i --anomaly ${BASE_PERIOD} --anomaly_freq month
