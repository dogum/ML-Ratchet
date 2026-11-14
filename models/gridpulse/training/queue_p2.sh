cd "$(dirname "$0")"   # training/ (sandbox: grid/gridpulse/src); logs and data in ../work/
while pgrep -f "bank.py train_p1b" > /dev/null; do sleep 20; done
python3 bank.py val_p2 p2 3000 98 0.6 > ../work/logs/bank_val_p2.log 2>&1
python3 bank.py train_p2 p2 24000 3 0.7 > ../work/logs/bank_train_p2.log 2>&1
