#!/bin/bash
# slot A: twin_dev -> val -> phys_exam -> train (worker 0 of 3)
cd "$(dirname "$0")"
./run_bank.sh "twin_dev 0 1 p1" "val 0 1 p1" "phys_exam 0 1 p1" "train 0 3 p1"
