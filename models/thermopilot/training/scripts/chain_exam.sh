#!/bin/sh
for s in $@; do python3 scripts/evaluate.py $s baselines; done
