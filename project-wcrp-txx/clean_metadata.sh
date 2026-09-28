#
# Bash script for cleaning up the metadata of supporting data files
#
# Usage: bash clean_metadata.sh {input_files} 
#
#

infiles=( $@ )


module load nco

for infile in "${infiles[@]}"; do

ncatted -O -h -a bounds,latitude,d,, ${infile}
ncatted -O -h -a bounds,longitude,d,, ${infile}
ncatted -O -h -a coordinates,gh500,d,, ${infile}
ncatted -O -h -a coordinates,mx2t,d,, ${infile}

done
