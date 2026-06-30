"""Create EC-Earth3 DCPP file lists"""

import os
import glob
import argparse

import numpy as np


file_dir = "/g/data/oi10/replicas/CMIP6/DCPP/EC-Earth-Consortium/EC-Earth3/dcppA-hindcast"
freq_dict = {
    'pr': 'day',
    'sfcWind': 'day',
    'tos': 'Omon'
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
    for year in np.arange(1960, 2017 + 1):
        infiles1 = glob.glob(f"{file_dir}/s{year}-r?i1p1f1/day/{var}/gr/v202012??/*.nc")
        infiles1.sort()
        infiles2 = glob.glob(f"{file_dir}/s{year}-r10i1p1f1/day/{var}/gr/v202012??/*.nc")
        infiles2.sort()
        infiles3 = glob.glob(f"{file_dir}/s{year}-r?i2p1f1/day/{var}/gr/*/*.nc")
        infiles3.sort()
        infiles4 = glob.glob(f"{file_dir}/s{year}-r10i2p1f1/day/{var}/gr/*/*.nc")
        infiles4.sort()
        infiles = infiles1 + infiles2 + infiles3 + infiles4
        assert len(infiles) == 165, f"year {year} pr does not have 165 (15*11) files"
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

