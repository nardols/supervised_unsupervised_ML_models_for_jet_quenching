#!/bin/bash

pt_label=$1
medium=$2

echo
echo "======================================================================================"
echo "   RUNNING TRANSFORMER TRAINING FOR pT RANGE: ${pt_label}, MEDIUM: ${medium}"
echo "======================================================================================"
echo

for scaler in off; do
    echo ">>> [TRAINING] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    python /eos/user/l/llimadas/ML_models/Transformers/TPE/train.py --scaler $scaler --medium $medium --pt_label $pt_label

    echo ">>> [TRAINING COMPLETED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    echo
done