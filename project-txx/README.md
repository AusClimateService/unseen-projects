# Annual maximum daily temperature (TXx) in Australia

The directory contains UNSEEN spatial analysis notebooks and associated configuration files related to extreme 1-day temperatures across Australia (Stellema et al., in review). 


## Work flow for generating the spatial analysis notebooks

```bash
# Regrid AGCD data
python regrid_files.py tmax /g/data/xv83/agcd-csiro/tmax/daily/tmax_AGCD-CSIRO_r005**.nc /g/data/xv83/unseen-projects/outputs/hazards/data/tmax_AGCD-CSIRO_r05_1901-2024.nc 0.5 conservative

# Define the project, obs and model details
DATA=BCC-CSM2-MR
PROJECT=txx
PROJECT_DETAILS=project-${PROJECT}/${PROJECT}_config.mk
OBS_DETAILS=dataset_makefiles/AGCD-CSIRO_r05_tasmax_config.mk

if [[ "$DATA" = "CAFE" ]]; then
   MODEL_DETAILS=dataset_makefiles/${DATA}_c5-d60-pX-f6_config.mk
elif [[ "$DATA" != "AGCD" ]]; then
   MODEL_DETAILS=dataset_makefiles/${DATA}_dcppA-hindcast_config.mk
fi

# Run the make commands
if [[ "$DATA" = "AGCD" ]]; then
   make metric-obs DATA=AGCD PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${OBS_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make obs-gev-params DATA=AGCD PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${OBS_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make metric-obs-spatial-analysis DATA=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${OBS_DETAILS} OBS_DETAILS=${OBS_DETAILS}
else
   make metric-forecast DATA=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${MODEL_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make forecast-diagnostics DATA=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${MODEL_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make forecast-gev-params DATA=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${MODEL_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make forecast-gev-params-additive-bc DATA=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${MODEL_DETAILS} OBS_DETAILS=${OBS_DETAILS}
   make metric-forecast-spatial-analysis MODEL=${DATA} PROJECT_DETAILS=${PROJECT_DETAILS} MODEL_DETAILS=${MODEL_DETAILS} OBS_DETAILS=${OBS_DETAILS}

# Multi-model plots
papermill -p metric txx -p obs_config_file AGCD-CSIRO_r05_tasmax_config.mk -p obs AGCD -p spatial_analysis_multimodel.ipynb project-txx/spatial_analysis_multimodel_txx_AGCD-CSIRO_r05.ipynb
papermill -p metric txx -p obs_config_file AGCD-CSIRO_r05_tasmax_config.mk -p obs AGCD -p bc additive spatial_analysis_multimodel.ipynb project-txx/spatial_analysis_multimodel_txx_AGCD-CSIRO_r05_bias-corrected-additive.ipynb
fi
```