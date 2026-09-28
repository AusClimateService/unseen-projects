"""Create CMCC-CM2-SR5 DCPP daily precipitation file list."""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/CMCC/CMCC-CM2-SR5/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon',
    'tasmax': 'day',
    'psl': 'day',
    'zg500': 'AERday',
}
nrun_dict = {
    'pr': 10,
    'sfcWind': 20,
    'tos': 10,
    'tasmax': 20,
    'psl': 10,
    'zg500': 20,
}


def create_file_list(var):
    """Create a file list"""

    outfile_name = f"CMCC-CM2-SR5_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    freq = freq_dict[var]
    version = '2021*' if var == 'tos' else "*" 
    nruns = nrun_dict[var]

    for year in np.arange(1960, 2020):
        infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/{freq}/{var}/gn/{version}/*.nc")
        infiles1.sort()
        infiles2 = glob.glob(f"{file_dir}/s{year}-r1?i1p1f1/{freq}/{var}/gn/{version}/*.nc")
        infiles2.sort()
        infiles3 = glob.glob(f"{file_dir}/s{year}-r20i1p1f1/{freq}/{var}/gn/{version}/*.nc")
        infiles3.sort()
        infiles = infiles1 + infiles2 + infiles3
        assert len(infiles) == nruns, f"year {year} does not have {nruns} {var} files"
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

