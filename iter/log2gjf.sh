#!/bin/bash
# Convert final Gaussian log geometries in the current folder to .gjf files using Multiwfn.

export PATH=$PATH:/opt/soft/Multiwfn_3.8_dev_bin_Linux

shopt -s nullglob
log_files=(*.log)
nfile=${#log_files[@]}

if [ "$nfile" -eq 0 ]; then
    echo "Warning: no .log files were found in $(pwd)."
    exit 0
fi

icc=0
for inf in "${log_files[@]}"; do
    ((icc++))
    outf="${inf%.log}.gjf"
    echo "Converting ${inf} to ${outf} ... (${icc} of ${nfile})"
    Multiwfn "${inf}" << EOF > /dev/null
100
2
10
${outf}
0
q
EOF
done
