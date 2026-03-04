#!/bin/bash

pt_label=$1
medium=$2

echo
echo "==========================================================================================="
echo "   RUNNING LSTM+ATT PREDICTION & EVALUATION FOR pT RANGE: ${pt_label}, MEDIUM: ${medium}   "
echo "==========================================================================================="
echo

for scaler in off; do
    echo ">>> [PREDICTION] medium=${medium}, scaler=${scaler}"
    python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/predict.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label
    echo ">>> [PREDICTION COMPLETED] medium=${medium}, scaler=${scaler}"

    python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/predict.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --inverter on
    echo ">>> [PREDICTION INVERTED COMPLETED] medium=${medium}, scaler=${scaler}"
    echo
done

for scaler in off; do
    echo ">>> [EVALUATION] medium=${medium}, scaler=${scaler}, threshold=0.5"
    python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/evaluate.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --threshold 0.5
    echo ">>> [EVALUATION COMPLETED] medium=${medium}, scaler=${scaler}, threshold=0.5"

    python /eos/user/l/llimadas/ML_models/LSTM+Att/TPE/evaluate.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --threshold 0.5 --inverter on
    echo ">>> [EVALUATION INVERTED COMPLETED] medium=${medium}, scaler=${scaler}, threshold=0.5"
    echo
done