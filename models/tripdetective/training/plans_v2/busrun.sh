#!/bin/bash
cd /c/Developer/ML-ratchet/models/tripdetective/training
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1
PY=../../../.venv/Scripts/python.exe
until grep -q 'bus chunk 29' ../work/logs/busbank_1.log; do sleep 20; done
$PY train.py p2_bus_train pb1 2500 feat=3 chdrop=0 > ../work/logs/pb1.train.log 2>&1
$PY busladder.py pb1.pt p2_bus_val "Bus network: CNN + set pooling + classical estimates on 1,500 simulated electric buses (power / 12)" > ../work/logs/pb1.ratchet.log 2>&1
$PY train.py p2_bus_train pb2 2500 feat=4 phys=1 chdrop=0 > ../work/logs/pb2.train.log 2>&1
$PY busladder.py pb2.pt p2_bus_val "+ physics-regression head (pooled road-load fit over the leg's windows) and the source-power channel" > ../work/logs/pb2.ratchet.log 2>&1
echo busrun done
