"""Script for producing a tier1 data file"""

import argparse
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import cmdline_provenance as cmdprov


rp_dict = {}
rp_dict['txx'] = {
    'BCC-CSM2-MR': 4649,
    'CanESM5': 1261,
    'CMCC-CM2-SR5': 7252962,
    'EC-Earth3': 2873332,
    'IPSL-CM6A-LR': 451265,
    'MIROC6': np.inf,
    'MPI-ESM1-2-HR': np.inf,
    'MRI-ESM2-0': np.inf,
    'NorCPM1': 30843,
}
rp_dict['rx1day'] = {
#    'BCC-CSM2-MR': ,
#    'CanESM5': ,
#    'CMCC-CM2-SR5': ,
#    'EC-Earth3': ,
#    'IPSL-CM6A-LR': ,
#    'MIROC6': ,
#    'MPI-ESM1-2-HR': ,
#    'MRI-ESM2-0': ,
#    'NorCPM1': 
}

percentiles = [5, 16, 84, 95]
rp_bnds_dict = {}
rp_bnds_dict['txx'] = {
    'BCC-CSM2-MR': (2238, 2882, 9219, 14842),
    'CanESM5': (910, 1022, 1589, 1836),
    'CMCC-CM2-SR5': (380159, 913310, 911513064, np.inf),
    'EC-Earth3': (167835, 384416, 1106837844, np.inf),
    'IPSL-CM6A-LR': (55890, 117219, 5309865, 197637458),
    'MIROC6': (np.inf, np.inf, np.inf, np.inf),
    'MPI-ESM1-2-HR': (np.inf, np.inf, np.inf, np.inf),
    'MRI-ESM2-0': (np.inf, np.inf, np.inf, np.inf),
    'NorCPM1': (13606, 18361, 58842, 94208),
}
rp_bnds_dict['rx1dy'] = {
#    'BCC-CSM2-MR': (),
#    'CanESM5': (),
#    'CMCC-CM2-SR5': (),
#    'EC-Earth3': (),
#    'IPSL-CM6A-LR': (),
#    'MIROC6': (),
#    'MPI-ESM1-2-HR': (),
#    'MRI-ESM2-0': (),
#    'NorCPM1': (),
}

sample_size_dict = {
    'BCC-CSM2-MR': 4320,
    'CanESM5': 9120,
    'CMCC-CM2-SR5': 8400,
    'EC-Earth3': 6960,
    'IPSL-CM6A-LR': 5130,
    'MIROC6': 4130,
    'MPI-ESM1-2-HR': 4720,
    'MRI-ESM2-0': 2400,
    'NorCPM1': 10620,
}


code_url = 'https://github.com/AusClimateService/unseen-projects/tree/master/project-wcrp-txx'
history_log = cmdprov.new_log(code_url=code_url)
global_attrs = {
    'institution': 'CSIRO',
    'method': 'UNSEEN',
    'event': 'PNW Heatwave',
    'data_tier': 'tier1',
    'project': 'WCRP Common Event Attribution and Assessment',
    'contact': 'damien.irving@csiro.au',
    'source': 'Decadal Climate Prediction Project (DCPP) models',
    'experiment': 'dcppA-hindcast',
    'references': 'Decadal Climate Prediction Project (DCPP): https://doi.org/10.5194/gmd-9-3751-2016. UNSEEN methodology: https://doi.org/10.1002/met.70118',
    'creation_date': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'history': history_log,
}


def processing_description(metric):
    """Describe the processing for a given metric"""
    
    if metric == 'txx':
        ref_year = 2021
        event = 'TXx value of 39.51 degC'
    
    intro = f'Extreme value analysis was conducted to calculate the return period (rp) for a {event} in the year {ref_year}.'
    rp1 = 'This involved fitting a non-stationary generalised extreme value (GEV) distribution to the model data using maximum likelihood estimation.'
    rp2 = f'The cumulative distribution function corresponding to the fitted GEV distribution for the year {ref_year} was then used to calculate rp.'
    uncert1 = 'The uncertainty bounds (rp_bnds) were determined by parametric bootstrapping, whereby the GEV distribution was repeatedly randomly sampled (with the same sample size as the model data) to obtain 1000 estimates of rp.'
    uncert2 = 'From these 1000 estimates the 5-95 percentile and 16-84 percentile uncertainty ranges could be calculated.'

    description = f'{intro} {rp1} {rp2} {uncert1} {uncert2}'
    
    return description
    

def get_encoding(ds):
    """Output file encoding."""

    encoding = {}
    ds_vars = list(ds.coords) + list(ds.keys())
    for ds_var in ds_vars:
        encoding[ds_var] = {'_FillValue': None}

    return encoding


def main(args):
    """Run the program"""

    models = list(rp_dict[args.metric].keys())
    rp_values = list(rp_dict[args.metric].values())
    rp_bnds_values = np.array(list(rp_bnds_dict[args.metric].values()))
    ss_values = list(sample_size_dict.values())

    rp_da = xr.DataArray(
        data=rp_values,
        dims=['model',],
        coords={'model': models,},
        name='rp',
        attrs = {
            'long_name': 'Return period',
            'units': 'years',
        }
    )
    rp_bnds_da = xr.DataArray(
        data=rp_bnds_values,
        dims=['model', 'percentile'],
        coords={'model': models, 'percentile': percentiles},
        name='rp_bnds',
        attrs = {
            'long_name': 'Return period uncertainty bounds',
            'units': 'years'
        }
    )
    ss_da = xr.DataArray(
        data=ss_values,
        dims=['model',],
        coords={'model': models,},
        name='n',
        attrs = {
            'long_name': 'Sample size',
            'units': '1',
        }
    )
    output_ds = xr.merge([rp_da, rp_bnds_da, ss_da])
    output_ds.attrs = global_attrs
    output_ds.attrs['processing'] = processing_description(args.metric)
    output_ds['model'].attrs['long_name'] = 'Model name'
    output_ds['percentile'].attrs['long_name'] = 'Percentile'
    output_ds['percentile'].attrs['units'] = '1'

    outdir = f'/g/data/xv83/unseen-projects/outputs/wcrp-{args.metric}/data'
    fpath = f'{outdir}/CSIRO_UNSEEN_tier1.nc'
    encoding = get_encoding(output_ds)
    output_ds.to_netcdf(fpath, encoding=encoding)
    print(fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )                      
    parser.add_argument("metric", type=str, choices=('txx', 'rx1day'), help="metric")           
    args = parser.parse_args()
    main(args)
