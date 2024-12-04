# unseen-projects

Code for running generic UNSEEN analysis workflows.

Individual UNSEEN project work can be found in the `project-*/` subdirectories.

## Standard usage

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
### Single location analysis
```bash
make metric-forecast-analysis MODEL=CanESM5 PROJECT_DETAILS=project-jasper/jasper_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/AGCD-precip_config.mk
```

### Spatial analysis
```bash
make metric-forecast-spatial-analysis MODEL=CanESM5 PROJECT_DETAILS=project-jasper/jasper_config.mk MODEL_DETAILS=dataset_makefiles/CanESM5_dcppA-hindcast_config.mk OBS_DETAILS=dataset_makefiles/AGCD-precip_config.mk
```

## Custom usage

To run the software in the UNSEEN package without using the standard `Makefile`,
run the following commands at the command line to activate a virtual environment
that has the UNSEEN package and all its dependencies installed.

```
module use /g/data/hh5/public/modules
module load conda/analysis3
source /g/data/xv83/unseen-projects/unseen_venv/bin/activate
```

(Run `deactivate` to exit the virtual environment.)
