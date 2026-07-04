#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/est_number_lib.sh" ]; then
    source "$SCRIPT_DIR/est_number_lib.sh"
else
    echo "ERROR: est_number_lib.sh not found in $SCRIPT_DIR" >&2
    exit 1
fi
load_est_numbers

POLL_INTERVAL=60

STATUS_FILE="momap_workflow_status.csv"
WATCH_KR_FILE="momap_kr_watch.list"
WATCH_KNR_FILE="momap_knr_watch.list"
WATCH_ISC_FILE="momap_isc_watch.list"

echo "molecule,status,message" > "$STATUS_FILE"
> "$WATCH_KR_FILE"
> "$WATCH_KNR_FILE"
> "$WATCH_ISC_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="$SCRIPT_DIR/MOMAP.slurm"

log_status() {
    mol="$1"
    status="$2"
    message="$3"

    echo "$mol,$status,$message" >> "$STATUS_FILE"
}

check_gaussian_normal() {
    log_file="$1"

    if [ ! -f "$log_file" ]; then
        return 1
    fi

    if tail -n 10 "$log_file" | grep -q "Normal termination of Gaussian"; then
        return 0
    else
        return 1
    fi
}

check_no_negative_frequency() {
    log_file="$1"

    if [ ! -f "$log_file" ]; then
        return 3
    fi

    awk '
    BEGIN {
        found_freq = 0
        has_negative = 0
    }

    /Frequencies --/ {
        found_freq = 1
        for (i = 3; i <= NF; i++) {
            if ($i + 0 < 0) {
                has_negative = 1
            }
        }
    }

    END {
        if (found_freq == 0) {
            exit 2
        }

        if (has_negative == 1) {
            exit 1
        }

        exit 0
    }
    ' "$log_file"

    return $?
}

check_orca_normal_in_soc() {
    soc_dir="$1"

    if [ ! -d "$soc_dir" ]; then
        return 1
    fi

    log_file="$soc_dir/log"

    if [ ! -f "$log_file" ]; then
        return 1
    fi

    if tail -n 10 "$log_file" | grep -q "ORCA TERMINATED NORMALLY"; then
        return 0
    else
        return 1
    fi
}

check_momap_done() {
    workdir="$1"

    if [ ! -d "$workdir" ]; then
        return 1
    fi

    found_out=0

    for out_file in "$workdir"/*_out.dat; do
        [ -f "$out_file" ] || continue
        found_out=1

        if grep -q "ALL SUCCESSFULLY DONE" "$out_file"; then
            return 0
        fi
    done

    if [ "$found_out" -eq 0 ]; then
        return 1
    fi

    return 1
}

job_is_active() {
    jobid="$1"

    if [ -z "$jobid" ]; then
        return 1
    fi

    if [ -n "$(squeue -j "$jobid" -h 2>/dev/null)" ]; then
        return 0
    else
        return 1
    fi
}

marker_job_active() {
    workdir="$1"
    label="$2"

    marker="$workdir/.${label}.jobid"

    if [ ! -f "$marker" ]; then
        return 1
    fi

    jobid=$(cat "$marker" | head -n 1)

    if job_is_active "$jobid"; then
        return 0
    else
        return 1
    fi
}

run_formchk_in_dir() {
    workdir="$1"
    mol="$2"
    tag="$3"

    if [ ! -d "$workdir" ]; then
        log_status "$mol" "failed" "$tag directory not found"
        return 1
    fi

    tadf_chk="$workdir/TADF.chk"

    if [ ! -f "$tadf_chk" ]; then
        log_status "$mol" "failed" "$tag TADF.chk not found"
        return 1
    fi

    expected_gjf="$workdir/${mol}-${tag}.gjf"

    if [ -f "$expected_gjf" ]; then
        gjf_file="$expected_gjf"
    else
        gjf_file=$(find "$workdir" -maxdepth 1 -type f -name "*.gjf" | sort | head -n 1)
    fi

    if [ -z "$gjf_file" ]; then
        log_status "$mol" "failed" "$tag gjf file not found"
        return 1
    fi

    base_name=$(basename "$gjf_file" .gjf)

    new_chk="$workdir/${base_name}.chk"
    new_fchk="$workdir/${base_name}.fchk"
    formchk_log="$workdir/formchk_${base_name}.log"

    cp -p "$tadf_chk" "$new_chk"

    rm -f "$new_fchk"

    formchk "$new_chk" "$new_fchk" > "$formchk_log" 2>&1

    if [ ! -s "$new_fchk" ]; then
        log_status "$mol" "failed" "$tag formchk failed"
        return 1
    fi

    return 0
}

copy_required_file() {
    src="$1"
    dst_dir="$2"
    mol="$3"
    label="$4"

    if [ ! -f "$src" ]; then
        log_status "$mol" "failed" "missing $label: $src"
        return 1
    fi

    mkdir -p "$dst_dir"
    cp -p "$src" "$dst_dir"/

    return 0
}

copy_slurm_file_from_template() {
    dst_dir="$1"
    mol="$2"
    label="$3"

    if [ ! -f "$SLURM_TEMPLATE" ]; then
        log_status "$mol" "failed" "MOMAP.slurm not found in script directory: $SLURM_TEMPLATE"
        return 1
    fi

    mkdir -p "$dst_dir"
    cp -p "$SLURM_TEMPLATE" "$dst_dir/MOMAP.slurm"

    if [ ! -f "$dst_dir/MOMAP.slurm" ]; then
        log_status "$mol" "failed" "failed to copy MOMAP.slurm to $label"
        return 1
    fi

    return 0
}

write_kr_evc_inp() {
    evc_dir="$1"
    mol="$2"

    cat > "$evc_dir/momap.inp" << EOF
do_evc            = 1

&evc
  ffreq(1)      = "${mol}-s0.log"
  ffreq(2)      = "${mol}-s1.log"
/
EOF
}

write_knr_evc_inp() {
    evc_dir="$1"
    mol="$2"

    cat > "$evc_dir/momap.inp" << EOF
do_evc            = 1

&evc
  ffreq(1)      = "${mol}-s0.log"
  ffreq(2)      = "${mol}-s1.log"
  fnacme        = "${mol}-nacme.log"
/
EOF
}

write_isc_evc_inp() {
    evc_dir="$1"
    mol="$2"

    cat > "$evc_dir/momap.inp" << EOF
do_evc            = 1

&evc
  ffreq(1)      = "${mol}-s1.log"
  ffreq(2)      = "${mol}-t103.log"
/
EOF
}

submit_job() {
    workdir="$1"
    mol="$2"
    label="$3"

    if [ ! -d "$workdir" ]; then
        log_status "$mol" "failed" "$label directory not found before sbatch"
        return 1
    fi

    if [ ! -f "$workdir/MOMAP.slurm" ]; then
        log_status "$mol" "failed" "$label MOMAP.slurm not found before sbatch"
        return 1
    fi

    if [ ! -f "$workdir/momap.inp" ]; then
        log_status "$mol" "failed" "$label momap.inp not found before sbatch"
        return 1
    fi

    if check_momap_done "$workdir"; then
        log_status "$mol" "skip" "$label already finished according to _out.dat"
        return 0
    fi

    if marker_job_active "$workdir" "$label"; then
        jobid=$(cat "$workdir/.${label}.jobid" | head -n 1)
        log_status "$mol" "skip" "$label already submitted and active: $jobid"
        return 0
    fi

    cd "$workdir" || return 1

    sbatch_output=$(sbatch MOMAP.slurm 2>&1)
    sbatch_status=$?

    cd - > /dev/null || return 1

    if [ "$sbatch_status" -ne 0 ]; then
        log_status "$mol" "failed" "$label sbatch failed: $sbatch_output"
        return 1
    fi

    jobid=$(echo "$sbatch_output" | awk '{print $NF}')
    echo "$jobid" > "$workdir/.${label}.jobid"

    log_status "$mol" "submitted" "$label submitted: $sbatch_output"
    return 0
}

extract_scf_energy() {
    log_file="$1"

    awk '/SCF Done/ {energy=$5} END {if (energy!="") print energy}' "$log_file"
}

extract_s0_energy() {
    s0_log="$1"

    extract_scf_energy "$s0_log"
}

extract_td_energy() {
    log_file="$1"

    awk '/Total Energy, E\(TD-HF\/TD-DFT\)/ {energy=$NF} END {if (energy!="") print energy}' "$log_file"
}

extract_s1_energy() {
    s1_log="$1"

    extract_td_energy "$s1_log"
}

extract_first_and_last_dips() {
    s1_log="$1"

    awk '
    /Ground to excited state transition electric dipole moments \(Au\):/ {
        in_block=1
        next
    }

    in_block && /^[[:space:]]*1[[:space:]]/ {
        if (first == "") {
            first = $5
        }
        last = $5
        in_block = 0
    }

    END {
        if (first != "" && last != "") {
            print first, last
        }
    }
    ' "$s1_log"
}

extract_hso_from_soc_log() {
    soc_log="$1"

    awk '
    /<T\|HSO\|S>/ {
        target_line = ""
        in_last_hso_block = 1
        next
    }

    in_last_hso_block && target_line == "" && /^[[:space:]]*1[[:space:]]+1[[:space:]]/ {
        target_line = $0
        in_last_hso_block = 0
        next
    }

    END {
        if (target_line == "") {
            exit 1
        }

        line = target_line
        n = 0

        while (match(line, /[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/)) {
            n++
            num[n] = substr(line, RSTART, RLENGTH)
            line = substr(line, RSTART + RLENGTH)
        }

        if (n < 8) {
            exit 1
        }

        sum = 0.0

        for (i = 3; i <= 8; i++) {
            sum += num[i] * num[i]
        }

        printf "%.10f", sqrt(sum)
    }
    ' "$soc_log"
}

is_number() {
    value="$1"

    awk -v x="$value" 'BEGIN {
        if (x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) exit 0;
        else exit 1;
    }'
}

extract_ead_from_evc_logs() {
    evc_dir="$1"
    mol="$2"

    s0_log="$evc_dir/${mol}-s0.log"
    s1_log="$evc_dir/${mol}-s1.log"

    if [ ! -s "$s0_log" ]; then
        return 1
    fi

    if [ ! -s "$s1_log" ]; then
        return 1
    fi

    s0_energy=$(extract_s0_energy "$s0_log")
    s1_energy=$(extract_s1_energy "$s1_log")

    if ! is_number "$s0_energy"; then
        return 1
    fi

    if ! is_number "$s1_energy"; then
        return 1
    fi

    awk -v s1="$s1_energy" -v s0="$s0_energy" 'BEGIN {printf "%.10f", s1 - s0}'
}

extract_isc_ead_from_logs() {
    isc_evc_dir="$1"
    mol="$2"

    s1_log="$isc_evc_dir/${mol}-s1.log"
    t103_log="$isc_evc_dir/${mol}-t103.log"

    if [ ! -s "$s1_log" ]; then
        return 1
    fi

    if [ ! -s "$t103_log" ]; then
        return 1
    fi

    s1_energy=$(extract_td_energy "$s1_log")
    t103_energy=$(extract_scf_energy "$t103_log")

    if ! is_number "$s1_energy"; then
        return 1
    fi

    if ! is_number "$t103_energy"; then
        return 1
    fi

    awk -v s1="$s1_energy" -v t103="$t103_energy" 'BEGIN {printf "%.10f", s1 - t103}'
}

prepare_kr_input_and_submit() {
    kr_evc_dir="$1"
    mol="$2"

    kr_dir="$kr_evc_dir/kr"
    mkdir -p "$kr_dir"

    if ! check_momap_done "$kr_evc_dir"; then
        log_status "$mol" "skip" "kr_evc _out.dat does not contain ALL SUCCESSFULLY DONE; kr not submitted"
        return 1
    fi

    if check_momap_done "$kr_dir"; then
        log_status "$mol" "skip" "kr already finished according to _out.dat"
        return 0
    fi

    if marker_job_active "$kr_dir" "kr"; then
        jobid=$(cat "$kr_dir/.kr.jobid" | head -n 1)
        log_status "$mol" "skip" "kr already submitted and active: $jobid"
        return 0
    fi

    evc_cart="$kr_evc_dir/evc.cart.dat"

    if [ ! -s "$evc_cart" ]; then
        log_status "$mol" "failed" "kr_evc finished but evc.cart.dat not found or empty"
        return 1
    fi

    cp -p "$evc_cart" "$kr_dir/evc.cart.dat"

    if [ ! -s "$kr_dir/evc.cart.dat" ]; then
        log_status "$mol" "failed" "failed to copy evc.cart.dat to kr directory"
        return 1
    fi

    if [ -f "$kr_evc_dir/MOMAP.slurm" ]; then
        cp -p "$kr_evc_dir/MOMAP.slurm" "$kr_dir/MOMAP.slurm"
    else
        copy_slurm_file_from_template "$kr_dir" "$mol" "kr" || return 1
    fi

    s0_log="$kr_evc_dir/${mol}-s0.log"
    s1_log="$kr_evc_dir/${mol}-s1.log"

    if [ ! -s "$s0_log" ]; then
        log_status "$mol" "failed" "s0 log not found in kr_evc for kr parameter extraction"
        return 1
    fi

    if [ ! -s "$s1_log" ]; then
        log_status "$mol" "failed" "s1 log not found in kr_evc for kr parameter extraction"
        return 1
    fi

    s0_energy=$(extract_s0_energy "$s0_log")
    s1_energy=$(extract_s1_energy "$s1_log")

    if ! is_number "$s0_energy"; then
        log_status "$mol" "failed" "failed to extract s0 energy from ${mol}-s0.log"
        return 1
    fi

    if ! is_number "$s1_energy"; then
        log_status "$mol" "failed" "failed to extract s1 energy from ${mol}-s1.log"
        return 1
    fi

    ead=$(awk -v s1="$s1_energy" -v s0="$s0_energy" 'BEGIN {printf "%.10f", s1 - s0}')

    dips_pair=$(extract_first_and_last_dips "$s1_log")
    dip_first=$(echo "$dips_pair" | awk '{print $1}')
    dip_last=$(echo "$dips_pair" | awk '{print $2}')

    if ! is_number "$dip_first"; then
        log_status "$mol" "failed" "failed to extract first Dip.S. from ${mol}-s1.log"
        return 1
    fi

    if ! is_number "$dip_last"; then
        log_status "$mol" "failed" "failed to extract last Dip.S. from ${mol}-s1.log"
        return 1
    fi

    edma=$(awk -v x="$dip_first" 'BEGIN {printf "%.10f", sqrt(x) * 2.5417}')
    edme=$(awk -v x="$dip_last"  'BEGIN {printf "%.10f", sqrt(x) * 2.5417}')

    cat > "$kr_dir/momap.inp" << EOF
do_spec_tvcf_ft    = 1
do_spec_tvcf_spec  = 1


&spec_tvcf
DUSHIN             =.t.
Temp               = 300 K
tmax               = 1000 fs
dt                 = 0.01 fs
Ead                = ${ead} au
EDMA               = ${edma} debye
EDME               = ${edme} debye
FreqScale          = 1.0
DSFile             = "evc.cart.dat"
Emax               = 0.3 au
dE                 = 0.00001 au
logFile            = "spec.tvcf.log"
FtFile             = "spec.tvcf.ft.dat"
FoFile             = "spec.tvcf.fo.dat"
FoSFile            = "spec.tvcf.spec.dat"
/
EOF

    log_status "$mol" "success" "kr input prepared: Ead=${ead} au EDMA=${edma} debye EDME=${edme} debye"

    submit_job "$kr_dir" "$mol" "kr"
    return $?
}

prepare_knr_input_and_submit() {
    knr_evc_dir="$1"
    mol="$2"

    knr_dir="$knr_evc_dir/knr"
    mkdir -p "$knr_dir"

    if ! check_momap_done "$knr_evc_dir"; then
        log_status "$mol" "skip" "knr_evc _out.dat does not contain ALL SUCCESSFULLY DONE; knr not submitted"
        return 1
    fi

    if check_momap_done "$knr_dir"; then
        log_status "$mol" "skip" "knr already finished according to _out.dat"
        return 0
    fi

    if marker_job_active "$knr_dir" "knr"; then
        jobid=$(cat "$knr_dir/.knr.jobid" | head -n 1)
        log_status "$mol" "skip" "knr already submitted and active: $jobid"
        return 0
    fi

    evc_cart="$knr_evc_dir/evc.cart.dat"
    evc_nac="$knr_evc_dir/evc.cart.nac"

    if [ ! -s "$evc_cart" ]; then
        log_status "$mol" "failed" "knr_evc finished but evc.cart.dat not found or empty"
        return 1
    fi

    if [ ! -s "$evc_nac" ]; then
        log_status "$mol" "failed" "knr_evc finished but evc.cart.nac not found or empty"
        return 1
    fi

    cp -p "$evc_cart" "$knr_dir/evc.cart.dat"
    cp -p "$evc_nac" "$knr_dir/evc.cart.nac"

    if [ ! -s "$knr_dir/evc.cart.dat" ]; then
        log_status "$mol" "failed" "failed to copy evc.cart.dat to knr directory"
        return 1
    fi

    if [ ! -s "$knr_dir/evc.cart.nac" ]; then
        log_status "$mol" "failed" "failed to copy evc.cart.nac to knr directory"
        return 1
    fi

    if [ -f "$knr_evc_dir/MOMAP.slurm" ]; then
        cp -p "$knr_evc_dir/MOMAP.slurm" "$knr_dir/MOMAP.slurm"
    else
        copy_slurm_file_from_template "$knr_dir" "$mol" "knr" || return 1
    fi

    ead=$(extract_ead_from_evc_logs "$knr_evc_dir" "$mol")

    if ! is_number "$ead"; then
        log_status "$mol" "failed" "failed to extract Ead for knr from ${mol}-s0.log and ${mol}-s1.log"
        return 1
    fi

    cat > "$knr_dir/momap.inp" << EOF
do_ic_tvcf_ft    = 1
do_ic_tvcf_spec  = 1


&ic_tvcf
DUSHIN             =.t.
Temp               = 300 K
tmax               = 1000 fs
dt                 = 0.01 fs
Ead                = ${ead} au
DSFile             = "evc.cart.dat"
CoulFile           = "evc.cart.nac"
Emax               = 0.3 au
logFile            = "ic.tvcf.log"
FtFile             = "ic.tvcf.ft.dat"
FoFile             = "ic.tvcf.fo.dat"
/
EOF

    log_status "$mol" "success" "knr input prepared: Ead=${ead} au"

    submit_job "$knr_dir" "$mol" "knr"
    return $?
}

prepare_isc_input_and_submit() {
    isc_evc_dir="$1"
    mol="$2"

    isc_dir="$isc_evc_dir/isc"
    mkdir -p "$isc_dir"

    if ! check_momap_done "$isc_evc_dir"; then
        log_status "$mol" "skip" "isc_evc _out.dat does not contain ALL SUCCESSFULLY DONE; isc not submitted"
        return 1
    fi

    if check_momap_done "$isc_dir"; then
        log_status "$mol" "skip" "isc already finished according to _out.dat"
        return 0
    fi

    if marker_job_active "$isc_dir" "isc"; then
        jobid=$(cat "$isc_dir/.isc.jobid" | head -n 1)
        log_status "$mol" "skip" "isc already submitted and active: $jobid"
        return 0
    fi

    evc_cart="$isc_evc_dir/evc.cart.dat"

    if [ ! -s "$evc_cart" ]; then
        log_status "$mol" "failed" "isc_evc finished but evc.cart.dat not found or empty"
        return 1
    fi

    cp -p "$evc_cart" "$isc_dir/evc.cart.dat"

    if [ ! -s "$isc_dir/evc.cart.dat" ]; then
        log_status "$mol" "failed" "failed to copy evc.cart.dat to isc directory"
        return 1
    fi

    if [ -f "$isc_evc_dir/MOMAP.slurm" ]; then
        cp -p "$isc_evc_dir/MOMAP.slurm" "$isc_dir/MOMAP.slurm"
    else
        copy_slurm_file_from_template "$isc_dir" "$mol" "isc" || return 1
    fi

    ead=$(extract_isc_ead_from_logs "$isc_evc_dir" "$mol")

    if ! is_number "$ead"; then
        log_status "$mol" "failed" "failed to extract ISC Ead from ${mol}-s1.log and ${mol}-t103.log"
        return 1
    fi

    soc_log="$mol/s0/s1/soc/log"

    if [ ! -s "$soc_log" ]; then
        log_status "$mol" "failed" "soc log not found for Hso extraction: $soc_log"
        return 1
    fi

    hso=$(extract_hso_from_soc_log "$soc_log")

    if ! is_number "$hso"; then
        log_status "$mol" "failed" "failed to extract Hso from soc log"
        return 1
    fi

    cat > "$isc_dir/momap.inp" << EOF
do_isc_tvcf_ft          = 1
do_isc_tvcf_spec        = 1

&isc_tvcf
DUSHIN                  = .t.
Temp                    = 298 K
tmax                    = 1500 fs
dt                      = 0.01 fs
Ead                     = ${ead} au
Hso                     = ${hso} cm-1
DSFile                  = "evc.cart.dat"
Emax                    = 0.3 au
logFile                 = "isc.tvcf.log"
FtFile                  = "isc.tvcf.ft.dat"
FoFile                  = "isc.tvcf.fo.dat"
/
EOF

    log_status "$mol" "success" "isc input prepared: Ead=${ead} au Hso=${hso} cm-1"

    submit_job "$isc_dir" "$mol" "isc"
    return $?
}

prepare_one_molecule() {
    mol="$1"

    echo "======================================"
    echo "Preparing $mol"
    echo "======================================"

    if [ ! -d "$mol" ]; then
        log_status "$mol" "skip" "est folder not found"
        return 1
    fi

    s0_dir="$mol/s0"
    s1_dir="$mol/s0/s1"
    t1_dir="$mol/s0/t103"
    nacme_dir="$mol/s0/nacme"
    soc_dir="$mol/s0/s1/soc"

    if [ ! -d "$s0_dir" ]; then
        log_status "$mol" "skip" "s0 folder not found"
        return 1
    fi

    if [ ! -d "$s1_dir" ]; then
        log_status "$mol" "skip" "s1 folder not found"
        return 1
    fi

    if [ ! -d "$t1_dir" ]; then
        log_status "$mol" "skip" "t103 folder not found"
        return 1
    fi

    if [ ! -d "$nacme_dir" ]; then
        log_status "$mol" "skip" "nacme folder not found"
        return 1
    fi

    if [ ! -d "$soc_dir" ]; then
        log_status "$mol" "skip" "soc folder not found"
        return 1
    fi

    if ! check_orca_normal_in_soc "$soc_dir"; then
        log_status "$mol" "skip" "soc ORCA not normally terminated"
        return 1
    fi

    t1_log="$t1_dir/${mol}-t103.log"
    nacme_log="$nacme_dir/${mol}-nacme.log"

    if ! check_gaussian_normal "$t1_log"; then
        log_status "$mol" "skip" "t103 Gaussian not normally terminated"
        return 1
    fi

    check_no_negative_frequency "$t1_log"
    freq_status=$?

    if [ "$freq_status" -eq 1 ]; then
        log_status "$mol" "skip" "t103 has imaginary frequency"
        return 1
    elif [ "$freq_status" -eq 2 ]; then
        log_status "$mol" "skip" "t103 frequency data not found"
        return 1
    elif [ "$freq_status" -eq 3 ]; then
        log_status "$mol" "skip" "t103 log file not found for frequency check"
        return 1
    elif [ "$freq_status" -ne 0 ]; then
        log_status "$mol" "skip" "t103 frequency check failed"
        return 1
    fi

    if ! check_gaussian_normal "$nacme_log"; then
        log_status "$mol" "skip" "nacme Gaussian not normally terminated"
        return 1
    fi

    run_formchk_in_dir "$s0_dir" "$mol" "s0" || return 1
    run_formchk_in_dir "$s1_dir" "$mol" "s1" || return 1
    run_formchk_in_dir "$t1_dir" "$mol" "t103" || return 1
    run_formchk_in_dir "$nacme_dir" "$mol" "nacme" || return 1

    s0_log="$s0_dir/${mol}-s0.log"
    s0_fchk="$s0_dir/${mol}-s0.fchk"

    s1_log="$s1_dir/${mol}-s1.log"
    s1_fchk="$s1_dir/${mol}-s1.fchk"

    t1_log="$t1_dir/${mol}-t103.log"
    t1_fchk="$t1_dir/${mol}-t103.fchk"

    nacme_log="$nacme_dir/${mol}-nacme.log"
    nacme_fchk="$nacme_dir/${mol}-nacme.fchk"

    kr_evc_dir="$s0_dir/momap/kr_evc"
    knr_evc_dir="$s0_dir/momap/knr_evc"
    isc_evc_dir="$s0_dir/momap/isc_evc"

    mkdir -p "$kr_evc_dir" "$knr_evc_dir" "$isc_evc_dir"
    mkdir -p "$kr_evc_dir/kr" "$knr_evc_dir/knr" "$isc_evc_dir/isc"

    copy_required_file "$s0_log" "$kr_evc_dir" "$mol" "s0 log to kr_evc" || return 1
    copy_required_file "$s0_fchk" "$kr_evc_dir" "$mol" "s0 fchk to kr_evc" || return 1
    copy_required_file "$s1_log" "$kr_evc_dir" "$mol" "s1 log to kr_evc" || return 1
    copy_required_file "$s1_fchk" "$kr_evc_dir" "$mol" "s1 fchk to kr_evc" || return 1

    copy_required_file "$s0_log" "$knr_evc_dir" "$mol" "s0 log to knr_evc" || return 1
    copy_required_file "$s0_fchk" "$knr_evc_dir" "$mol" "s0 fchk to knr_evc" || return 1
    copy_required_file "$s1_log" "$knr_evc_dir" "$mol" "s1 log to knr_evc" || return 1
    copy_required_file "$s1_fchk" "$knr_evc_dir" "$mol" "s1 fchk to knr_evc" || return 1
    copy_required_file "$nacme_log" "$knr_evc_dir" "$mol" "nacme log to knr_evc" || return 1
    copy_required_file "$nacme_fchk" "$knr_evc_dir" "$mol" "nacme fchk to knr_evc" || return 1

    copy_required_file "$s1_log" "$isc_evc_dir" "$mol" "s1 log to isc_evc" || return 1
    copy_required_file "$s1_fchk" "$isc_evc_dir" "$mol" "s1 fchk to isc_evc" || return 1
    copy_required_file "$t1_log" "$isc_evc_dir" "$mol" "t103 log to isc_evc" || return 1
    copy_required_file "$t1_fchk" "$isc_evc_dir" "$mol" "t103 fchk to isc_evc" || return 1

    copy_slurm_file_from_template "$kr_evc_dir" "$mol" "kr_evc" || return 1
    copy_slurm_file_from_template "$knr_evc_dir" "$mol" "knr_evc" || return 1
    copy_slurm_file_from_template "$isc_evc_dir" "$mol" "isc_evc" || return 1

    write_kr_evc_inp "$kr_evc_dir" "$mol"
    write_knr_evc_inp "$knr_evc_dir" "$mol"
    write_isc_evc_inp "$isc_evc_dir" "$mol"

    log_status "$mol" "success" "files and EVC inputs prepared"

    submit_job "$kr_evc_dir" "$mol" "kr_evc"
    submit_job "$knr_evc_dir" "$mol" "knr_evc"
    submit_job "$isc_evc_dir" "$mol" "isc_evc"

    if check_momap_done "$kr_evc_dir"; then
        prepare_kr_input_and_submit "$kr_evc_dir" "$mol"
    else
        echo "$mol|$kr_evc_dir" >> "$WATCH_KR_FILE"
        log_status "$mol" "watching" "kr_evc not finished yet; added to kr monitor list"
    fi

    if check_momap_done "$knr_evc_dir"; then
        prepare_knr_input_and_submit "$knr_evc_dir" "$mol"
    else
        echo "$mol|$knr_evc_dir" >> "$WATCH_KNR_FILE"
        log_status "$mol" "watching" "knr_evc not finished yet; added to knr monitor list"
    fi

    if check_momap_done "$isc_evc_dir"; then
        prepare_isc_input_and_submit "$isc_evc_dir" "$mol"
    else
        echo "$mol|$isc_evc_dir" >> "$WATCH_ISC_FILE"
        log_status "$mol" "watching" "isc_evc not finished yet; added to isc monitor list"
    fi

    return 0
}

process_kr_watch_file() {
    if [ ! -s "$WATCH_KR_FILE" ]; then
        return 0
    fi

    tmp_watch="${WATCH_KR_FILE}.tmp"
    > "$tmp_watch"

    while IFS="|" read -r mol kr_evc_dir; do
        [ -n "$mol" ] || continue
        [ -n "$kr_evc_dir" ] || continue

        kr_dir="$kr_evc_dir/kr"

        if check_momap_done "$kr_dir"; then
            log_status "$mol" "skip" "kr already finished; removed from kr watch list"
            continue
        fi

        if marker_job_active "$kr_dir" "kr"; then
            jobid=$(cat "$kr_dir/.kr.jobid" | head -n 1)
            log_status "$mol" "skip" "kr already active: $jobid; removed from kr watch list"
            continue
        fi

        if check_momap_done "$kr_evc_dir"; then
            echo "$mol : kr_evc finished. Preparing kr..."
            prepare_kr_input_and_submit "$kr_evc_dir" "$mol"
            continue
        fi

        if marker_job_active "$kr_evc_dir" "kr_evc"; then
            echo "$mol : kr_evc still running or pending."
            echo "$mol|$kr_evc_dir" >> "$tmp_watch"
        else
            log_status "$mol" "failed" "kr_evc job is not active and _out.dat has no ALL SUCCESSFULLY DONE"
        fi

    done < "$WATCH_KR_FILE"

    mv "$tmp_watch" "$WATCH_KR_FILE"
}

process_knr_watch_file() {
    if [ ! -s "$WATCH_KNR_FILE" ]; then
        return 0
    fi

    tmp_watch="${WATCH_KNR_FILE}.tmp"
    > "$tmp_watch"

    while IFS="|" read -r mol knr_evc_dir; do
        [ -n "$mol" ] || continue
        [ -n "$knr_evc_dir" ] || continue

        knr_dir="$knr_evc_dir/knr"

        if check_momap_done "$knr_dir"; then
            log_status "$mol" "skip" "knr already finished; removed from knr watch list"
            continue
        fi

        if marker_job_active "$knr_dir" "knr"; then
            jobid=$(cat "$knr_dir/.knr.jobid" | head -n 1)
            log_status "$mol" "skip" "knr already active: $jobid; removed from knr watch list"
            continue
        fi

        if check_momap_done "$knr_evc_dir"; then
            echo "$mol : knr_evc finished. Preparing knr..."
            prepare_knr_input_and_submit "$knr_evc_dir" "$mol"
            continue
        fi

        if marker_job_active "$knr_evc_dir" "knr_evc"; then
            echo "$mol : knr_evc still running or pending."
            echo "$mol|$knr_evc_dir" >> "$tmp_watch"
        else
            log_status "$mol" "failed" "knr_evc job is not active and _out.dat has no ALL SUCCESSFULLY DONE"
        fi

    done < "$WATCH_KNR_FILE"

    mv "$tmp_watch" "$WATCH_KNR_FILE"
}

process_isc_watch_file() {
    if [ ! -s "$WATCH_ISC_FILE" ]; then
        return 0
    fi

    tmp_watch="${WATCH_ISC_FILE}.tmp"
    > "$tmp_watch"

    while IFS="|" read -r mol isc_evc_dir; do
        [ -n "$mol" ] || continue
        [ -n "$isc_evc_dir" ] || continue

        isc_dir="$isc_evc_dir/isc"

        if check_momap_done "$isc_dir"; then
            log_status "$mol" "skip" "isc already finished; removed from isc watch list"
            continue
        fi

        if marker_job_active "$isc_dir" "isc"; then
            jobid=$(cat "$isc_dir/.isc.jobid" | head -n 1)
            log_status "$mol" "skip" "isc already active: $jobid; removed from isc watch list"
            continue
        fi

        if check_momap_done "$isc_evc_dir"; then
            echo "$mol : isc_evc finished. Preparing isc..."
            prepare_isc_input_and_submit "$isc_evc_dir" "$mol"
            continue
        fi

        if marker_job_active "$isc_evc_dir" "isc_evc"; then
            echo "$mol : isc_evc still running or pending."
            echo "$mol|$isc_evc_dir" >> "$tmp_watch"
        else
            log_status "$mol" "failed" "isc_evc job is not active and _out.dat has no ALL SUCCESSFULLY DONE"
        fi

    done < "$WATCH_ISC_FILE"

    mv "$tmp_watch" "$WATCH_ISC_FILE"
}

monitor_evc_and_submit_rates() {
    if [ ! -s "$WATCH_KR_FILE" ] && [ ! -s "$WATCH_KNR_FILE" ] && [ ! -s "$WATCH_ISC_FILE" ]; then
        echo "No kr_evc, knr_evc or isc_evc jobs need monitoring."
        return 0
    fi

    echo "Start monitoring kr_evc, knr_evc and isc_evc jobs..."
    echo "Polling interval: ${POLL_INTERVAL} s"

    while [ -s "$WATCH_KR_FILE" ] || [ -s "$WATCH_KNR_FILE" ] || [ -s "$WATCH_ISC_FILE" ]; do
        process_kr_watch_file
        process_knr_watch_file
        process_isc_watch_file

        if [ -s "$WATCH_KR_FILE" ] || [ -s "$WATCH_KNR_FILE" ] || [ -s "$WATCH_ISC_FILE" ]; then
            sleep "$POLL_INTERVAL"
        fi
    done

    echo "All kr_evc, knr_evc and isc_evc watch-list jobs have been processed."
}

if [ ! -f "$SLURM_TEMPLATE" ]; then
    echo "ERROR: MOMAP.slurm not found in script directory: $SLURM_TEMPLATE"
    echo "Please put MOMAP.slurm, momap.sh and momap-sbatch.sh in the same folder."
    exit 1
fi

for i in "${EST_NUMBERS[@]}"; do
    mol="est-$i"
    prepare_one_molecule "$mol"
done

monitor_evc_and_submit_rates

echo "Finished."
echo "Status file: $STATUS_FILE"





