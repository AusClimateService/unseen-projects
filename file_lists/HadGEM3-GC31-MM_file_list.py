"""Create HadGEM3-GC31-MM DCPP file list."""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast"
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

    outfile_name = f"HadGEM3-GC31-MM_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass
    start_year = 1963 if var == 'sfcWind' else 1960
    freq = freq_dict[var]
    for year in np.arange(start_year, 2018 + 1):
        infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f2/{freq}/{var}/gn/v20200417/*.nc")
        infiles1.sort()
        infiles2 = glob.glob(f"{file_dir}/s{year}-r1?i1p1f2/{freq}/{var}/gn/v20200417/*.nc")
        infiles2.sort()
        infiles = infiles1 + infiles2
        assert len(infiles) == 120, f"year {year} does not have 120 {var} files"
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

