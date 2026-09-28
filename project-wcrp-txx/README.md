## WCRP TXx

The [common event attribution and assessment (CEAA) project](https://www.wcrp-climate.org/epesc-wg3/epesc-wg3-ceaa)
of the WCRP Explaining and Predicting Earth System Change (EPESC) Lighthouse Activity
seeks to identify extreme weather and climate events for coordinated study.

This repository contains the analysis related to our submission
to the 2021 Pacific Northwest Heatwave case study.
The [guidance note](https://docs.google.com/document/d/1Ww_oW_ynmPco3cbG18hO9fdoFwH6VA7HPRyXGiCUryU/edit?usp=sharing)
explains the submission requirements.

### Data processing

Step 1: Calculate the TXx metric from observations:  
```
make metric-obs PROJECT_DETAILS=project-wcrp-txx/txx_config.mk OBS_DETAILS=dataset_makefiles/ERA5-tasmax_config.mk
```

Step 2: Calculate the TXx metric for a model:  
```
make metric-forecast MODEL=CanESM5 PROJECT_DETAILS=project-wcrp-txx/txx_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/ERA5-tasmax_config.mk
```

Step 3: Calculate the annual tasmax for a model:  
```
make metric-forecast MODEL=MRI-ESM2-0 PROJECT_DETAILS=project-wcrp-txx/config_clim_tasmax.mk MODEL_DETAILS=dataset_makefiles/MRI-ESM2-0_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/ERA5-tasmax_config.mk
```

Step 4: Copy a `txx_*.ipynb` notebook and run it for that model.  


### Data availability

The guidance documents ask for the following variables:
- 2m temperature: `tasmax`
- Z500 geopotential height: `z500`
- Mean sea level pressure: `psl`
- Total cloud cover: `clt`
- Soil moisture in the top layer: `mrsos`
- Total atmospheric column water vapour: `prw`

The availability of each variable is listed below:  
:green_circle: = data is available on NCI  
:yellow_circle: = data is available on ESGF  
:white_circle: = data not available  

| model | tasmax | psl | z500 | clt | prw | mrsos |
| ---   | :-:    | :-: | :-:  | :-: | :-: | :-:   |
| BCC-CSM2-MR | :green_circle: | :green_circle: | :yellow_circle: | :white_circle: | :white_circle: | :white_circle: |
| CanESM5 | :green_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| CMCC-CM2-SR5 | :green_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| EC-Earth3 | :green_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| IPSL-CM6A-LR | :green_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| MIROC6 | :green_circle: | :green_circle: | :green_circle: (zg) | :white_circle: | :white_circle: | :white_circle: |
| MPI-ESM1-2-HR | :green_circle: | :green_circle: | :yellow_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| MRI-ESM2-0 | :green_circle: | :green_circle: | :white_circle: | :yellow_circle: | :white_circle: | :white_circle: |
| NorCPM1 | :green_circle: | :green_circle: | :yellow_circle: | :white_circle: | :yellow_circle: | :white_circle: |


