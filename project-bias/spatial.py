import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import geopandas as gpd
import regionmask
import xarray as xr

from unseen import fileio

import utils


model_dict = {
    'BCC-CSM2-MR': 'tab:blue',
    'CAFE': 'tab:orange',
    'CMCC-CM2-SR5': 'tab:green',
    'CanESM5': 'tab:red',
    'EC-Earth3': 'tab:purple',
    'IPSL-CM6A-LR': 'tab:brown',
    'MIROC6': 'tab:pink',
    'MPI-ESM1-2-HR': 'tab:grey',
    'MRI-ESM2-0': 'tab:olive',
    'NorCPM1': 'tab:cyan',
}


def extract_closest_row(df, return_period):
    """Extract the row from df closest to the return period."""

    differences = np.array(np.abs(df.index - return_period))
    index = np.argmin(differences)

    return df.iloc[index]


def get_mask():
    """Get ocean mask."""

    infile = '/g/data/xv83/unseen-projects/outputs/bias/data/rx1day_AGCD-CSIRO_1901-2024_annual-jul-to-jun_AUS300i.nc'
    ds = fileio.open_dataset(infile)
    overlap_threshold = 0.67
    shape_gpd = gpd.read_file('/g/data/ia39/aus-ref-clim-data-nci/shapefiles/data/australia/australia.shp')
    shape_rgm = regionmask.from_geopandas(
        shape_gpd,
        names="AUS_NAME21",
        abbrevs="AUS_CODE21",
        name="australia"
    )
    frac = shape_rgm.mask_3D_frac_approx(ds)
    mask = frac.sel(region=0) >= overlap_threshold

    return mask


def main(args):
    """Run the program."""

    mask = get_mask()

    return_periods = np.array([100, 1000])
    nlevels = len(return_periods)
    nlats, nlons = mask.shape
    G2_array = np.zeros([nlevels, nlats, nlons])
    M2_array = np.zeros([nlevels, nlats, nlons])
    B2_array = np.zeros([nlevels, nlats, nlons])
    T2_array = np.zeros([nlevels, nlats, nlons])
    TMB2_array = np.zeros([nlevels, nlats, nlons])
    OG2_array = np.zeros([nlevels, nlats, nlons])
    OM2_array = np.zeros([nlevels, nlats, nlons])
    OT2_array = np.zeros([nlevels, nlats, nlons])
    MMM_array = np.zeros([nlevels, nlats, nlons])
    obs_array = np.zeros([nlevels, nlats, nlons])
    nmodels_array = np.zeros([nlats, nlons])
    for lat_index in range(nlats):
        for lon_index in range(nlons):
            logging.info(f'lat index: {lat_index}, lon index: {lon_index}')

            include_point = mask.isel({'lat': lat_index, 'lon': lon_index})
            if include_point:
                return_df, gev_spread_df = utils.get_return_values(
                    args.metric,
                    [lat_index, lon_index],
                    model_dict,
                    similarity_check=True
                )
                nmodels = return_df.filter(like='model-bc-mean').shape[1]
            else:
                nmodels = np.nan
            nmodels_array[lat_index, lon_index] = nmodels            

            if nmodels >= 2:
                obs, MMM, uncertainty = utils.uncertainty_breakdown(return_df, gev_spread_df)
                G2, M2, B2, T2, TMB2, OG2, OM2, OT2 = uncertainty
                for level_index, return_period in enumerate(return_periods):
                    G2s = extract_closest_row(G2, return_period)
                    M2s = extract_closest_row(M2, return_period)
                    B2s = extract_closest_row(B2, return_period)
                    T2s = extract_closest_row(T2, return_period)
                    TMB2s = extract_closest_row(TMB2, return_period)
                    OG2s = extract_closest_row(OG2, return_period)
                    OM2s = extract_closest_row(OM2, return_period)
                    OT2s = extract_closest_row(OT2, return_period)
                    MMMs = extract_closest_row(MMM, return_period)
                    obss = extract_closest_row(obs, return_period)
                    G2_array[level_index, lat_index, lon_index] = G2s
                    M2_array[level_index, lat_index, lon_index] = M2s
                    B2_array[level_index, lat_index, lon_index] = B2s
                    T2_array[level_index, lat_index, lon_index] = T2s
                    TMB2_array[level_index, lat_index, lon_index] = TMB2s
                    OG2_array[level_index, lat_index, lon_index] = OG2s
                    OM2_array[level_index, lat_index, lon_index] = OM2s
                    OT2_array[level_index, lat_index, lon_index] = OT2s
                    MMM_array[level_index, lat_index, lon_index] = MMMs
                    obs_array[level_index, lat_index, lon_index] = obss
            elif nmodels in [0, 1]:
                obs, uncertainty = utils.obs_uncertainty_breakdown(return_df, gev_spread_df)
                OG2, OM2, OT2 = uncertainty
                for level_index in range(nlevels):
                    OG2s = extract_closest_row(OG2, return_period)
                    OM2s = extract_closest_row(OM2, return_period)
                    OT2s = extract_closest_row(OT2, return_period)
                    obss = extract_closest_row(obs, return_period)
                    G2_array[level_index, lat_index, lon_index] = np.nan
                    M2_array[level_index, lat_index, lon_index] = np.nan
                    B2_array[level_index, lat_index, lon_index] = np.nan
                    T2_array[level_index, lat_index, lon_index] = np.nan
                    TMB2_array[level_index, lat_index, lon_index] = np.nan
                    OG2_array[level_index, lat_index, lon_index] = OG2s
                    OM2_array[level_index, lat_index, lon_index] = OM2s
                    OT2_array[level_index, lat_index, lon_index] = OT2s
                    MMM_array[level_index, lat_index, lon_index] = np.nan
                    obs_array[level_index, lat_index, lon_index] = obss
            else:                    
                for level_index in range(nlevels):
                    G2_array[level_index, lat_index, lon_index] = np.nan
                    M2_array[level_index, lat_index, lon_index] = np.nan
                    B2_array[level_index, lat_index, lon_index] = np.nan
                    T2_array[level_index, lat_index, lon_index] = np.nan
                    TMB2_array[level_index, lat_index, lon_index] = np.nan
                    OG2_array[level_index, lat_index, lon_index] = np.nan
                    OM2_array[level_index, lat_index, lon_index] = np.nan
                    OT2_array[level_index, lat_index, lon_index] = np.nan
                    MMM_array[level_index, lat_index, lon_index] = np.nan
                    obs_array[level_index, lat_index, lon_index] = np.nan

    units_dict = {'txx': 'Celsius', 'rx1day': 'mm'}
    units = units_dict[args.metric]
    ds_out = xr.Dataset(
        data_vars={
            'G2': (['lev', 'lat', 'lon'], G2_array, {'long_name': 'EVA uncertainty (model)', 'units': units}),
            'M2': (['lev', 'lat', 'lon'], M2_array, {'long_name': 'dataset uncertainty (model)', 'units': units}),
            'B2': (['lev', 'lat', 'lon'], B2_array, {'long_name': 'bias correction uncertainty (model)', 'units': units}),
            'T2': (['lev', 'lat', 'lon'], T2_array, {'long_name': 'total model uncertainty', 'units': units}),
            'TMB2': (['lev', 'lat', 'lon'], TMB2_array, {'long_name': 'combined dataset and bias uncertainty (model)', 'units': units}),
            'OG2': (['lev', 'lat', 'lon'], OG2_array, {'long_name': 'EVA uncertainty (AGCD)', 'units': units}),
            'OM2': (['lev', 'lat', 'lon'], OM2_array, {'long_name': 'dataset uncertainty (obs)', 'units': units}),
            'OT2': (['lev', 'lat', 'lon'], OT2_array, {'long_name': 'total observations uncertainty', 'units': units}),
            'MMM': (['lev', 'lat', 'lon'], MMM_array, {'long_name': 'multi-model mean (mean correction)', 'units': units}),
            'obs': (['lev', 'lat', 'lon'], obs_array, {'long_name': 'observations (AGCD)', 'units': units}),
            'nmodels': (['lat', 'lon'], nmodels_array, {'long_name': 'number of models', 'units': ' '}),
        },
        coords={
            'lev': (['lev',], return_periods, {'standard_name': 'return_period', 'long_name': 'return period', 'units': 'years'}),
            'lat': (['lat',], mask.lat.values, {'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}),
            'lon': (['lon',], mask.lon.values, {'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}),
        },
        attrs={
            'metric': args.metric,
        },
    )
    ds_out.to_netcdf(args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )     
    parser.add_argument("metric", type=str, choices=('txx', 'rx1day'), help="metric")
    parser.add_argument("outfile", type=str, help="output file name")
    args = parser.parse_args()
    main(args)
