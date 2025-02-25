# Configuration file for Makefile workflows

MODEL=IPSL-CM6A-LR
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1970-01-01 2017-12-31
BASE_PERIOD_TEXT=1970-2017
TIME_PERIOD_TEXT=196101-201701
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 10 --metadata_file /g/data/xv83/unseen-projects/code/dataset_config/dataset_dcpp.yml
MODEL_NINO_OPTIONS=--n_ensemble_files 10 --lon_bnds -170 -120 --lat_dim nav_lat --lon_dim nav_lon --agg_y_dim y --agg_x_dim x --anomaly ${BASE_PERIOD} --anomaly_freq month
