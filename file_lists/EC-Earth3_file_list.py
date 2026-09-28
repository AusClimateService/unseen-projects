"""Create EC-Earth3 DCPP file lists"""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon',
    'tasmax': 'day',
    'psl': 'day',
    'zg500': 'AERday',
}
nfiles_dict = {
    'pr': 165,  # 15*11
    'sfcWind': 99, # 9*11
    'tos': 110,  # 10*11
    'tasmax': 165,
    'psl': 165,
    'zg500': 165,
}


def create_file_list(var):
    """Create a file list"""

    outfile_name = f"EC-Earth3_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    # From 2005-2010 there are no r1i4p1f1 files, so i4 has been left out
    # 2018 is left out because it does not have i2 files
    # sfcWind is missing s1976-r8i1p1f1 so I've left out r8
    runs = [1, 2, 3, 4, 5, 6, 7, 9, 10] if var == 'sfcWind' else np.arange(1, 11)
    freq = freq_dict[var]
    nfiles = nfiles_dict[var]
    for year in np.arange(1960, 2017 + 1):
        infiles = []
        for run in runs:
            infiles_i1 = glob.glob(f"{file_dir}/s{year}-r{run}i1p1f1/{freq}/{var}/gr/v202012??/*.nc")
            infiles_i1.sort()
            if var in ['sfcWind', 'tos']:
                infiles_i2 = []
            else:
                infiles_i2 = glob.glob(f"{file_dir}/s{year}-r{run}i2p1f1/{freq}/{var}/gr/*/*.nc")
                infiles_i2.sort()
            infiles = infiles + infiles_i1 + infiles_i2
        assert len(infiles) == nfiles, f"year {year} has {len(infiles)} of {nfiles} {var} files"
        #if len(infiles) != nfiles:
        #    print(f"year {year} has {len(infiles)} of {nfiles} {var} files")
        with open(outfile_name, "a") as outfile:
            for item in infiles:
                outfile.write(f"{item}\n")


def main(args):
    """Run the program"""

    for var in args.vars:
        create_file_list(var)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vars", type=str, nargs='*', help="Variables to process")
    args = parser.parse_args()
    main(args)

