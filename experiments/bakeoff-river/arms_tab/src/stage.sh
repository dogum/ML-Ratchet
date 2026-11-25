#!/bin/bash
# stage.sh ARM RUNG : copy code + the rung's weights into the arm folder (what predict.py reads)
set -e
A=/home/claude/bakeoff/river/arms_tab; S=$A/src
case $1 in
 gbm) cp $S/feats.py $S/treeexport.py $A/tab_gbm/; cp $S/predict_gbm.py $A/tab_gbm/predict.py
      cp $A/tab_gbm/work/rungs/$2/gbm_trees.npz $A/tab_gbm/ ;;
 gp)  cp $S/feats.py $A/tab_gp/; cp $S/predict_gp.py $A/tab_gp/predict.py; cp $A/tab_gp/work/rungs/$2/gp.npz $A/tab_gp/ ;;
 dpl) cp $S/wsim.py $S/physarm.py $A/tab_dpl/; cp $S/predict_dpl.py $A/tab_dpl/predict.py
      cp $A/tab_dpl/work/rungs/$2/dpl_weights.npz $A/tab_dpl/ ;;
 formula) cp $S/wsim.py $S/physarm.py $S/lawsearch.py $S/formula_eval.py $A/tab_formula/
      cp $S/predict_formula.py $A/tab_formula/predict.py
      cp $A/tab_formula/work/rungs/$2/formulas.json $A/tab_formula/work/rungs/$2/bands.npz $A/tab_formula/ ;;
esac
echo staged $1 $2
