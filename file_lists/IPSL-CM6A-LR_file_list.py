"""Create IPSL-CM6A-LR DCPP file lists"""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon',
    'tasmax': 'day',
    'psl': 'day',
    'zg500': 'AERday'
}


def create_file_list(var):
    """Create a file list"""

    outfile_name = f"IPSL-CM6A-LR_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    freq = freq_dict[var]
    grid = 'gn' if var == 'tos' else 'gr'

    for year in np.arange(1960, 2016 + 1):
        infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/{freq}/{var}/{grid}/v20200108/*.nc")
        infiles1.sort()
        infiles2 = glob.glob(f"{file_dir}/s{year}-r??i1p1f1/{freq}/{var}/{grid}/v20200108/*.nc")
        infiles2.sort()
        infiles = infiles1 + infiles2
        assert len(infiles) == 10, f"year {year} has {len(infiles)} of 10 {var} files"
        #if len(infiles) != 10:
        #    print(f"year {year} has {len(infiles)} of 10 {var} files")
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

