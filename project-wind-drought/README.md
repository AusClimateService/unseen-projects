## Wind drought analysis

The wind drought analysis involves calculating the spatially averaged daily mean surface wind speed
for four different wind energy zones (NEM_2030, SE_2030, NWIS, SWIS) across the months of 
the year with the least sunlight hours (May - July).

This calculation is performed for observations (BARRA-R2)
and a number of models from the Decadal Climate Prediction Project (DCPP).

Once those daily timescale data files have been generated,
they can be used to calculate the annual maximum wind drought duration.

### Step 1: Generate a separate shapefile for each region

The [unseen software](https://github.com/AusClimateService/unseen) cannot handle overlapping shapefiles,
so each of the four regions of interest must be processed separately.
The `shapes.ipynb` notebook creates separate shapefiles for each region.

The resulting shapefiles are archived at `/g/data/xv83/unseen-projects/outputs/wind-drought/shapefiles`.

### Step 2: Generate spatially averaged daily mean surface wind speed files

Running `make` with the `metric-obs` (for BARRA-R2) or `metric-forecast` (for a DCPP model of interest) target
will run the `fileio` command line program to generate the files:

```
make <target> MODEL=CanESM5 PROJECT_DETAILS=project-wind-drought/wind-drought_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/BARRA-sfcWind_config.mk
```

The resulting data files are archived at `/g/data/xv83/unseen-projects/outputs/wind-drought/data`.

### Step 3: Calculate and analyse the annual maximum wind drought duration

This is done via a separate notebook for each model and region (e.g. `analysis_CanESM5_nem-2030.ipynb`).