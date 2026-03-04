#!/bin/bash

pt_label=$1
medium=$2

echo
echo "==============================================================================================="
echo "   RUNNING TRANSFORMER PREDICTION & EVALUATION FOR pT RANGE: ${pt_label}, MEDIUM: ${medium}    "
echo "==============================================================================================="
echo

for scaler in off; do

    # ------------ prediction ------------ #

    echo ">>> [PREDICTION] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    python /eos/user/l/llimadas/ML_models/Transformers/TPE/predict.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label
    echo ">>> [PREDICTION COMPLETED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    echo

    echo ">>> [PREDICTION INVERTED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    python /eos/user/l/llimadas/ML_models/Transformers/TPE/predict.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --inverter on
    echo ">>> [PREDICTION INVERTED COMPLETED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    echo


    # ------------ evaluation ------------ #
    

    echo ">>> [EVALUATION] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    python /eos/user/l/llimadas/ML_models/Transformers/TPE/evaluate.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --threshold 0.5
    echo ">>> [EVALUATION COMPLETED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    echo

    echo ">>> [EVALUATION INVERTED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    python /eos/user/l/llimadas/ML_models/Transformers/TPE/evaluate.py --medium $medium --scaler $scaler --mode test --pt_label $pt_label --threshold 0.5 --inverter on
    echo ">>> [EVALUATION INVERTED COMPLETED] medium=${medium}, scaler=${scaler}, pT=${pt_label}"
    echo
done