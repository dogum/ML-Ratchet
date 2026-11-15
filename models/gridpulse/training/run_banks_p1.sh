cd "$(dirname "$0")"   # training/ (sandbox: grid/gridpulse/src); logs and data in ../work/
export OMP_NUM_THREADS=1
nice -n 5 python3 bank.py val_p1 p1 3000 99 0.6 > ../work/logs/bank_val_p1.log 2>&1
nice -n 5 python3 bank.py train_p1 p1 24000 1 0.6 > ../work/logs/bank_train_p1.log 2>&1
