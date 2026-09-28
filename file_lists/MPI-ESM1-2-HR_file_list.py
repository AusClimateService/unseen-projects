"""Create MPI-ESM1-2-HR DCPP file list."""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/MPI-M/MPI-ESM1-2-HR/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon',
    'tasmax': 'day',
    'psl': 'day',
    'zg500': 'AERday',
}


def create_file_list(var):
    """Create a file list"""

    outfile_name = f"MPI-ESM1-2-HR_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    freq = freq_dict[var]
    nruns = 5 if var == 'tos' else 10
    # 2019 left out because it only has 5 of the 10 runs
    for year in np.arange(1960, 2018 + 1):
        infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/{freq}/{var}/gn/*/*.nc")
        infiles1.sort()
        infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/{freq}/{var}/gn/*/*.nc")
        infiles2.sort()
        infiles = infiles1 + infiles2
        assert len(infiles) == nruns, f"year {year} has {len(infiles)} of {nruns} {var} files"
#        if len(infiles) != nruns:
#            print(f"year {year} has {len(infiles)} of {nruns} {var} files")
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

