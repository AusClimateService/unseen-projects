## Wind drought analysis

To generate the daily data for percentile calculation (used by `percentile.ipynb`)
run the following with the `metric-obs` or `metric-forecast` target.

```
make <target> MODEL=CanESM5 PROJECT_DETAILS=project-wind-drought/percentile_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/BARRA-sfcWind_config.mk
```
