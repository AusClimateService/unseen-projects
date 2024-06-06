# UNSEEN analysis of growing season rainfall (GSR) events in Western and South Australian low cropping regions

## Notebooks
#### `prepare_shapefile.ipynb`  
Used to generate the shapefiles corresponding to the study regions (used by `Makefile`).

#### `analysis_{model-name}.ipynb`
Presents the analysis of each model.
Created by creating a file list using the relevant script in the `file_lists/` directory and then the `Makefile`.
See [README.md](https://github.com/AusClimateService/unseen-projects/blob/master/README.md).

#### `obs.ipynb`
Plots Australian Gridded Climate Data (AGCD) GSR timeseries and autocorrelation
in the WA and SA regions.

## Python scripts

These scripts require first saving the AGCD and DCPP model Apr-Oct rainfall (global grid and over the SA and WA regions).

For example,
```/envs/unseen2/bin/filio /g/data/zv2/agcd/v2-0-1/precip/total/r005/01month/agcd_v2-0-1_precip_total_r005_monthly*.nc --variables pr --months 4 5 6 7 8 8 10 --time_freq A-DEC --time_agg sum --verbose --metadata_file dataset_config/dataset_agcd_monthly.yml --input_freq M```

#### `process_gsr_data.py`
Returns xarray datasets of GSR variables `pr`, `decile` and `tercile`. The function `gsr_data_regions` returns GSR data averaged over the WA and SA regions (stacked along dimension `x`). The `gsr_data_aus_AGCD` and `gsr_data_aus_DCPP` functions returns GSR variables on their native grid subset to Australia south of around 25S.

#### `gsr_events.py`
Contains functions to define n-year GSR events and calculate transition probabilities. GSR events are based on consecutive years of low or high decile/tercile GSR. Note that transition probabilities use a modified event event definition that includes overlapping years.

#### `plot_gsr_regions.py`
Contains functions to plot event statistics and transition probabilities for the WA and SA regions.

#### `plot_gsr_maps.py`
Contains functions to plot spatial maps of event statistics and transition probabilities.

#### `utils.py`
Contains functions to calculate confidence intervals and combine figures.

#### `run_gsr_plot.py`
Runs all of the plot functions in `plot_gsr_regions.py` and `plot_gsr_maps.py` for each model.

## Software environment

The command line programs used by the Makefile come from our [UNSEEN package](https://github.com/AusClimateService/unseen).

The conda environment corresponds to `environment.yml`.

