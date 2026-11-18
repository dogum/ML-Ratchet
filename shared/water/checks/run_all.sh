#!/bin/bash
# Re-run every WaterSim check (single thread, niced). Order: R references, analytic, balance, speed, generator,
# check_weather and check_prior_skill read the dev calibrations of a previous run;
# then the dev calibrations (~25 min each objective; resumable: delete out/dev_calib* to redo) and SNOTEL.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
REFENV="${WATERSIM_REFENV:-$(cd .. && pwd)/watersim/.refenv}"   # isolated R env (not in git; see watersim/README.md)
if [ -x $REFENV/mm/bin/micromamba ]; then
  export MAMBA_ROOT_PREFIX=$REFENV/mamba
  (cd ref && $REFENV/mm/bin/micromamba run -p $REFENV/renv Rscript airgr_run.R > ../out/airgr.log 2>&1 \
          && $REFENV/mm/bin/micromamba run -p $REFENV/renv Rscript tuw_run.R > ../out/tuw.log 2>&1)
fi
for s in check_reference.py check_analytic.py check_balance.py check_speed.py check_generator.py check_weather.py check_reservoir.py check_prior_skill.py; do
  echo "== $s"; nice -n 5 "${PYTHON:-python3}" "$s" > "out/${s%.py}.log" 2>&1; tail -1 "out/${s%.py}.log"
done
nice -n 5 "${PYTHON:-python3}" calibrate_dev.py > out/calibrate_dev.log 2>&1
OBJ=nse nice -n 5 "${PYTHON:-python3}" calibrate_dev.py > out/calibrate_dev_nse.log 2>&1
nice -n 5 "${PYTHON:-python3}" summarize_dev.py > out/summarize_dev.log 2>&1
OBJ=nse nice -n 5 "${PYTHON:-python3}" summarize_dev.py >> out/summarize_dev.log 2>&1
[ -f check_snotel.py ] && nice -n 5 "${PYTHON:-python3}" check_snotel.py > out/check_snotel.log 2>&1
echo done
