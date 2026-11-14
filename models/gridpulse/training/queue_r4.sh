cd "$(dirname "$0")"   # training/ (sandbox: grid/gridpulse/src); logs and data in ../work/
while [ $(ls ../work/data/bank_train_p1b | wc -l) -lt 120 ]; do sleep 15; done
sleep 5
python3 train.py r4 train_p1,train_p1b mean 0 80 '{"ctx": true}' > ../work/logs/train_r4.log 2>&1
python3 train.py r5 train_p1,train_p1b mean 0 120 '{"ctx": true, "hid": [512, 512, 256]}' > ../work/logs/train_r5.log 2>&1
