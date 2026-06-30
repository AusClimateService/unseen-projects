#!/bin/bash
#PBS -P xv83
#PBS -q normal
#PBS -l walltime=4:00:00
#PBS -l mem=50GB
#PBS -l storage=gdata/xv83+gdata/ob53
#PBS -l wd
#PBS -v region

# Example 1: qsub -v region=nem-2030 barra_job.sh
# Example 2: qsub -v region=se-2030 barra_job.sh
# Example 3: qsub -v region=nwis barra_job.sh
# Example 4: qsub -v region=swis barra_job.sh

# Note: Run cdo mergetime afterwards to merge output into a single file


__conda_setup="$('/g/data/xv83/dbi599/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/g/data/xv83/dbi599/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/g/data/xv83/dbi599/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/g/data/xv83/dbi599/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate unseen

for year in $(seq 1980 2025); do
command="fileio /g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/day/sfcWind/latest/sfcWind_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_day_${year}*.nc /g/data/xv83/unseen-projects/outputs/wind-drought/data/sfcWind_BARRA-R2_${year}_MJJ_${region}.nc --variables sfcWind --months 5 6 7 --shapefile /g/data/xv83/unseen-projects/outputs/wind-drought/shapefiles/${region}.shp --shp_overlap 0.1 --spatial_agg weighted_mean --verbose --lat_bnds -45 -10 --lon_bnds 110 160"
echo ${command}
${command}
done



