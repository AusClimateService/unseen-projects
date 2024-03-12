# unseen-projects

Code for running generic UNSEEN analysis.

Individual UNSEEN project work can be found in the `project-*/` subdirectories.

## Usage

Step 1: Create a `project-{name}/` directory for your project.

Step 2: Create a configuration file in that project directory (i.e. `{name}_config.mk`)

Step 3: Run the makefile for a given model and observational dataset.

In the first instance, a job needs to be submitted to calculate the metric of interest in the forecast data:
```bash
make metric-forecast MODEL=CanESM5 PROJECT_DETAILS=project-jasper/jasper_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/AGCD-precip_config.mk
```

Next, the independence test needs to be run to decide the minimum lead time that can be retained:
```bash
make independence-test MODEL=CanESM5 PROJECT_DETAILS=project-jasper/jasper_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/AGCD-precip_config.mk
```

Once the minimum lead time is identified, the remainder of the analysis can be processed:
```bash
make metric-forecast-analysis MODEL=CanESM5 MIN_LEAD=0 PROJECT_DETAILS=project-jasper/jasper_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/AGCD-precip_config.mk
```
