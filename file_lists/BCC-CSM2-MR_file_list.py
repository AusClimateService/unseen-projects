"""Create BCC-CSM2-MR DCPP file lists"""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/BCC/BCC-CSM2-MR/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon'
}


def create_file_list(var):
    """Create a file list"""

    outfile_name = f"BCC-CSM2-MR_dcppA-hindcast_{var}_files.txt"
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    for year in np.arange(1961, 2014 + 1):
        infiles = glob.glob(f"{file_dir}/s{year}-r*i1p1f1/day/{var}/gn/*/*.nc")
        infiles.sort()
        assert len(infiles) == 8, f"year {year} pr does not have 8 files"
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

