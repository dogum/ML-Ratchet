#!/bin/sh
# bank chain (one process at a time): frozen sets first, then training banks
cd "$(dirname "$0")"
while pgrep -f "python3 noise_bank.py" > /dev/null; do sleep 20; done
python3 simbank.py p0 val_p0 3000 777 val > ../logs/bank_val_p0.log 2>&1
python3 simbank.py p1 val_p1 3000 778 val > ../logs/bank_val_p1.log 2>&1
python3 simbank.py p0 train_p0 40000 1 train > ../logs/bank_train_p0.log 2>&1
python3 simbank.py p1 train_p1 40000 2 train > ../logs/bank_train_p1.log 2>&1
