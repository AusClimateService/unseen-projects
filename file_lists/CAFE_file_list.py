"""Create CAFE-f6 file lists."""

import os
import glob

# Precipitation (pr)
pr_file = "CAFE_c5-d60-pX-f6_pr_files_nov-starts.txt"
try:
    os.remove(pr_file)
except OSError:
    pass

infiles_1990s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]11*/atmos_isobaric_daily.zarr.zip"
)
infiles_1990s.sort()
infiles_2000s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2???11*/atmos_isobaric_daily.zarr.zip"
)
infiles_2000s.sort()
with open(pr_file, "a") as outfile:
    for item in infiles_1990s + infiles_2000s:
        outfile.write(f"{item}\n")

# # Oceanic temperature at the surface (tos)
# tos_file = "CAFE_c5-d60-pX-f6_tos_files.txt"
# try:
#    os.remove(tos_file)
# except OSError:
#    pass
# infiles_1990s = glob.glob(
#    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/ocean_month.zarr.zip"
# )
# infiles_1990s.sort()
# infiles_2000s = glob.glob(
#    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2*/ocean_month.zarr.zip"
# )
# infiles_2000s.sort()
# with open(tos_file, "a") as outfile:
#    for item in infiles_1990s + infiles_2000s:
#        outfile.write(f"{item}\n")

# Daily maximum atmospheric temperature at the surface (tasmax)
tasmax_file = "CAFE_c5-d60-pX-f6_tasmax_files.txt"
try:
    os.remove(tasmax_file)
except OSError:
    pass
infiles_1990s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/atmos_hybrid_daily.zarr.zip"
)
infiles_1990s.sort()
infiles_2000s = glob.glob(
    "/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2???*/atmos_hybrid_daily.zarr.zip"
)
infiles_2000s.sort()
with open(tasmax_file, "a") as outfile:
    for item in infiles_1990s + infiles_2000s:
        outfile.write(f"{item}\n")

# Tasmax (separate May and Nov starts)
for m, tasmax_file in zip(
    ["05", "11"],
    [
        "CAFE_c5-d60-pX-f6_tasmax_files_may-starts.txt",
        "CAFE_c5-d60-pX-f6_tasmax_files_nov-starts.txt",
    ],
):
    try:
        os.remove(tasmax_file)
    except OSError:
        pass
    infiles_1990s = glob.glob(
        f"/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]{m}*/atmos_hybrid_daily.zarr.zip"
    )
    infiles_1990s.sort()
    infiles_2000s = glob.glob(
        f"/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2???{m}*/atmos_hybrid_daily.zarr.zip"
    )
    infiles_2000s.sort()
    with open(tasmax_file, "a") as outfile:
        for item in infiles_1990s + infiles_2000s:
            outfile.write(f"{item}\n")
